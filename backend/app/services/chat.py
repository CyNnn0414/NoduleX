from __future__ import annotations

from collections import Counter
import re

import httpx

from ..config import Settings
from ..models import ChatMessage, NoduleFinding, StudyRecord


class ChatService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def answer_patient_question(self, study: StudyRecord, question: str) -> tuple[str, list[str]]:
        context = self._build_patient_context(study)
        citations = [f"Nodule {finding.id}" for finding in study.patient_visible_findings()]

        if self.settings.openai_api_key:
            answer = await self._call_openai(study=study, context=context, question=question)
        else:
            answer = self._fallback_answer(study=study, question=question)

        study.patient_chat_history.append(ChatMessage(role="user", content=question))
        study.patient_chat_history.append(ChatMessage(role="assistant", content=answer))
        return answer, citations

    def _build_patient_context(self, study: StudyRecord) -> str:
        visible = study.patient_visible_findings()
        if not visible:
            finding_lines = ["No approved nodules are currently visible in this study."]
        else:
            finding_lines = [
                (
                    f"{index}. Slice {finding.slice_index}, {finding.classification}, risk {finding.malignancy_risk}, "
                    f"largest diameter {finding.measurement.longest_diameter_mm:.1f} mm. "
                    f"Patient summary: {finding.patient_summary}"
                )
                for index, finding in enumerate(visible, start=1)
            ]

        return "\n".join(
            [
                f"Patient name: {study.patient.name}",
                f"Age: {study.patient.age}",
                f"Smoking history: {study.patient.smoking_history}",
                f"Basic diagnosis: {study.basic_diagnosis or 'No AI screening summary is available yet.'}",
                "Approved findings:",
                *finding_lines,
                "Assistant instructions: answer in plain language, stay grounded in the study, and do not give treatment orders.",
            ]
        )

    def _build_system_prompt(self, context: str) -> str:
        return (
            "You are a patient education assistant for lung nodules. "
            "Use only the provided patient context and basic general lung nodule education. "
            "Answer in plain language, adapt directly to the patient's question, avoid sounding repetitive, "
            "do not give treatment orders, and do not provide extra study details unless the patient asks for them. "
            "If the patient asks a narrow question like count, size, or risk, answer that question briefly in 1-2 sentences.\n\n"
            f"Patient context:\n{context}"
        )

    def _build_message_history(self, study: StudyRecord) -> list[dict[str, str]]:
        return [
            {"role": turn.role, "content": turn.content}
            for turn in study.patient_chat_history[-self.settings.patient_chat_history_limit :]
            if turn.role in {"user", "assistant"} and turn.content.strip()
        ]

    def _extract_openai_output_text(self, data: dict) -> str | None:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
        return None

    async def _call_openai(self, study: StudyRecord, context: str, question: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.patient_chat_model,
            "instructions": self._build_system_prompt(context),
            "input": [
                *self._build_message_history(study),
                {"role": "user", "content": question},
            ],
            "temperature": self.settings.patient_chat_temperature,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.settings.openai_base_url.rstrip('/')}/responses",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                detail = error.response.text.strip() or error.response.reason_phrase
                raise RuntimeError(f"OpenAI request failed: {detail}") from error
            except httpx.HTTPError as error:
                raise RuntimeError("Unable to reach OpenAI for patient chat.") from error

        answer = self._extract_openai_output_text(response.json())
        if answer:
            return answer
        raise RuntimeError("OpenAI returned an empty patient chat response.")

    def _fallback_answer(self, study: StudyRecord, question: str) -> str:
        lowered = question.lower()
        findings = study.patient_visible_findings()
        detected_findings = study.findings
        history_seed = len([turn for turn in study.patient_chat_history if turn.role == "assistant"])
        opener = self._pick_opener(question=question, seed=history_seed)

        if self._contains_word(lowered, "hello", "hi", "hey"):
            return (
                f"{opener} I can help explain what this scan shows, how many nodules were marked, their sizes, "
                "and what terms like ground-glass or solid usually mean in plain language."
            )

        if self._contains_word(lowered, "thank", "thanks"):
            return (
                f"{opener} If you want, you can ask about the number of nodules, the largest one, "
                "the risk wording, or what a specific nodule type usually means."
            )

        if not findings:
            return (
                "I do not see any doctor-accepted nodules available for patient view in this study right now."
            )

        if self._matches(lowered, "accepted by the doctor", "doctor accepted", "accepted nodules", "approved by the doctor"):
            return (
                f"The doctor has accepted {len(findings)} nodule"
                f"{'' if len(findings) == 1 else 's'} for patient view."
            )

        if self._matches(lowered, "how many", "count", "number of nodules", "how much nodules", "how many nodules", "were found", "detected"):
            return (
                f"The AI detected {len(detected_findings)} nodule"
                f"{'' if len(detected_findings) == 1 else 's'} in this scan, and the doctor has accepted "
                f"{len(findings)} for patient view."
            )

        if self._matches(lowered, "what is a lung nodule", "what does a lung nodule mean", "what do lung nodules mean", "what is lung nodule"):
            return (
                "A lung nodule is a small spot seen in the lung on a scan. "
                "Many nodules are not cancer, but doctors look at the size and appearance to decide what it may mean."
            )

        if self._matches(lowered, "size", "big", "largest", "smallest", "diameter", "mm"):
            largest = self._largest_finding(findings)
            return (
                f"The largest doctor-accepted nodule is about {largest.measurement.longest_diameter_mm:.1f} mm "
                f"and is described as {largest.classification}."
            )

        if self._matches(lowered, "risk", "dangerous", "serious", "cancer", "malignant", "malignancy"):
            highest_risk = self._highest_risk(findings)
            largest = self._largest_finding(findings)
            return (
                f"The highest screening label in this study is {highest_risk} risk. "
                "This is a screening summary, not a confirmed diagnosis."
            )

        if self._matches(lowered, "ground-glass", "solid", "part-solid", "classification", "type", "kind"):
            return (
                "These words describe how a nodule looks on the scan. "
                "Ground-glass means a hazier-looking spot, solid means a denser spot, and part-solid means it has features of both."
            )

        if self._matches(lowered, "diagnosis", "impression", "summary", "what does the scan show", "what does this mean"):
            return (
                f"{study.basic_diagnosis or 'No AI screening summary is available yet.'} "
                "A clinician still needs to confirm what these findings mean for you personally."
            )

        if self._matches(lowered, "smoking", "smoker", "history"):
            return (
                f"The patient profile linked to this study says: {study.patient.smoking_history}. "
                "Smoking history can matter when doctors interpret lung nodules, but it is only one part of the overall picture."
            )

        if self._matches(lowered, "follow up", "next step", "next steps", "what should i do", "should i worry"):
            return (
                "I can explain what the scan summary says, but I should not tell you what medical action to take. "
                f"Based on the current review, {study.basic_diagnosis or 'there is an AI screening summary for your clinician to review.'} "
                "The best next step is to discuss the result with your clinician or radiologist, who can interpret it in context."
            )

        largest = self._largest_finding(findings)
        return (
            f"{opener} This study currently shows {len(findings)} doctor-accepted nodule"
            f"{'' if len(findings) == 1 else 's'}, and the largest is about "
            f"{largest.measurement.longest_diameter_mm:.1f} mm. "
            "If you want, ask me about the count, sizes, risk wording, or what a specific nodule type usually means."
        )

    def _matches(self, lowered_question: str, *phrases: str) -> bool:
        return any(phrase in lowered_question for phrase in phrases)

    def _contains_word(self, lowered_question: str, *words: str) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", lowered_question) for word in words)

    def _largest_finding(self, findings: list[NoduleFinding]) -> NoduleFinding:
        return max(findings, key=lambda finding: finding.measurement.longest_diameter_mm)

    def _highest_risk(self, findings: list[NoduleFinding]) -> str:
        if any(finding.malignancy_risk == "high" for finding in findings):
            return "high"
        if any(finding.malignancy_risk == "intermediate" for finding in findings):
            return "intermediate"
        return "low"

    def _classification_sentence(self, findings: list[NoduleFinding]) -> str:
        counts = Counter(finding.classification for finding in findings)
        parts = [f"{count} {classification}" for classification, count in counts.items()]
        if len(parts) == 1:
            return f"The approved findings are {parts[0]}."
        if len(parts) == 2:
            return f"The approved findings include {parts[0]} and {parts[1]}."
        return f"The approved findings include {', '.join(parts[:-1])}, and {parts[-1]}."

    def _pick_opener(self, question: str, seed: int) -> str:
        lowered = question.lower()
        if "?" in question:
            families = [
                "Based on this scan,",
                "From the reviewed findings,",
                "Looking at the current study,",
            ]
        elif self._matches(lowered, "how many", "count"):
            families = [
                "From the current review,",
                "Based on the approved findings,",
                "Looking at this study,",
            ]
        else:
            families = [
                "Here is the plain-language version.",
                "Here is what I can tell from this study.",
                "This is how I would explain it simply.",
            ]
        return families[seed % len(families)]
