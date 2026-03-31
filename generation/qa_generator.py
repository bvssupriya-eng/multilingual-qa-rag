import os
from llama_cpp import Llama
from config import MISTRAL_MODEL_PATH
import atexit

class QAGenerator:

    def __init__(self):
        # Check if model file exists
        if not os.path.exists(MISTRAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Mistral model not found at: {MISTRAL_MODEL_PATH}\n"
                f"Please download the model and place it in the models/ directory."
            )
        
        print("Loading Mistral GGUF model...")

        self.model = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            verbose=False
        )

        atexit.register(self.model.close)

    def _generate_base_answer(self, question, context, language="en"):
        """Step 1: Generate complete factual answer in English"""
        
        prompt = f"""[CONTEXT]
{context[:3000]}

[QUESTION]
{question}

[TASK]
Answer the question completely using the given context.

[IMPORTANT RULES]
- ONLY answer if the context is relevant to the question
- If the context does NOT contain information about the question topic, respond EXACTLY: "The provided context does not contain relevant information to answer this question."
- Use ALL relevant information from the context
- Do not give partial answers
- Do not skip important points
- Include all key facts and details
- Write a full, coherent answer (not short responses)
- If specific information is missing, say "Not found in context"
- Do NOT add information not present in the context
- Do NOT answer from general knowledge

[ANSWER]
"""

        output = self.model(
            prompt,
            max_tokens=400,  # Increased for complete answers
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.2,
        )

        return output["choices"][0]["text"].strip()

    def _apply_role_style(self, answer, role, language="en"):
        """Step 2: Apply role-based styling without losing information"""
        
        if role == "beginner":
            style_instruction = "Rewrite in simple, everyday language suitable for beginners. Use easy words and short sentences. Keep ALL the information."
        elif role == "teacher":
            style_instruction = "Rewrite in formal academic language suitable for professors. Use proper terminology and structured format. Keep ALL the information."
        elif role == "eval":
            # For eval, return as-is (already factual)
            return answer
        else:  # student
            style_instruction = "Rewrite in clear educational language suitable for students. Balance clarity with proper terminology. Keep ALL the information."

        prompt = f"""[TASK]
Rewrite the following answer in the specified style.

[CRITICAL RULES]
- Do NOT remove any information
- Do NOT shorten the answer
- Only change the writing style and tone
- Keep all facts, details, and examples
- Maintain completeness

[STYLE]
{style_instruction}

[ORIGINAL ANSWER]
{answer}

[REWRITTEN ANSWER]
"""

        output = self.model(
            prompt,
            max_tokens=450,
            temperature=0.25,
            top_p=0.9,
            repeat_penalty=1.2,
        )

        return output["choices"][0]["text"].strip()

    def generate_answer(self, question, context, role="student", language="en", extra_instruction=""):
        """Main generation function with 2-step process"""
        
        # Step 1: Generate complete factual answer in English
        base_answer = self._generate_base_answer(question, context, language)
        
        # Handle "Not found" case
        if "not found" in base_answer.lower()[:50]:
            return base_answer
        
        # Step 2: Apply role styling (only if not eval mode)
        if role == "eval":
            final_answer = base_answer
        else:
            final_answer = self._apply_role_style(base_answer, role, language)
        
        # Handle extra instructions (for regeneration)
        if extra_instruction:
            prompt = f"""[TASK]
Refine the following answer based on additional instructions.

[ADDITIONAL INSTRUCTIONS]
{extra_instruction}

[ORIGINAL ANSWER]
{final_answer}

[REFINED ANSWER]
"""
            output = self.model(
                prompt,
                max_tokens=400,
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.2,
            )
            final_answer = output["choices"][0]["text"].strip()
        
        return final_answer
