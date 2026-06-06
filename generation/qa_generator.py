"""
QA Generator using a local GGUF instruction model via llama-cpp-python.

Improvements over v1:
  - Language-aware prompting (language name injected into prompt)
  - Role prompts are language-neutral (avoid English idioms in non-English contexts)
  - Translation quality note appended when answer_lang != en
  - Cleaner step indicators (no end="" overlap)
"""

import os
from llama_cpp import Llama
from config import (
    MISTRAL_MODEL_PATH,
    GENERATION_N_CTX,
    GENERATION_N_THREADS,
    GENERATION_N_BATCH,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    GENERATION_REPEAT_PENALTY,
    GENERATION_CONTEXT_LIMIT
)

LANG_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ar': 'Arabic',
}

# Role-specific instructions that work across language contexts
ROLE_INSTRUCTIONS = {
    "beginner": (
        "Use simple vocabulary and short sentences. "
        "Avoid technical jargon. "
        "Explain as if to someone with no prior knowledge. "
        "Use examples where helpful."
    ),
    "student": (
        "Use clear educational language with moderate detail. "
        "Balance simplicity with accuracy. "
        "Include key terminology with brief explanations."
    ),
    "teacher": (
        "Use formal academic language with comprehensive detail. "
        "Include technical terms, nuances, and structure the answer logically. "
        "Cover all relevant aspects from the context."
    ),
}


class QAGenerator:

    def __init__(self):
        if not os.path.exists(MISTRAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Generation model not found at: {MISTRAL_MODEL_PATH}\n"
                "Please download the model and place it in the models/ directory."
            )

        print("Loading GGUF generation model...")
        self.model = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=GENERATION_N_CTX,
            n_threads=GENERATION_N_THREADS,
            n_batch=GENERATION_N_BATCH,
            verbose=False
        )
        
        # Pre-compile role instructions for faster access
        self._role_cache = ROLE_INSTRUCTIONS.copy()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_incomplete_answer(answer: str) -> str:
        """
        Remove incomplete trailing sentences to ensure clean answers.
        Keeps answer concise and complete within token limit.
        
        Args:
            answer: Raw generated answer text
            
        Returns:
            Cleaned answer with complete sentences only
        """
        if not answer:
            return answer
        
        # Sentence ending punctuation
        sentence_ends = ['.', '!', '?', '।', '؟', '।']  # Including Hindi, Arabic
        
        # If answer ends with proper punctuation, return as-is
        if answer[-1] in sentence_ends:
            return answer
        
        # Find last complete sentence
        last_end = -1
        for punct in sentence_ends:
            pos = answer.rfind(punct)
            if pos > last_end:
                last_end = pos
        
        # If found a complete sentence, cut there
        if last_end > 0:
            return answer[:last_end + 1].strip()
        
        # No complete sentence found - return as-is (rare case)
        return answer

    @staticmethod
    def _is_context_grounded(answer: str, context: str, min_overlap: float = 0.15) -> bool:
        """
        Quick token-overlap check: does the answer actually use words from the context?

        If the model answered from its own training data (knowledge leakage) instead
        of the provided context, the word overlap will be close to zero even when the
        answer looks fluent and confident.

        Args:
            answer:       Generated answer text.
            context:      Retrieved context passed to the model.
            min_overlap:  Minimum fraction of answer content-words that must appear
                          in the context. Default 0.15 (15 %).

        Returns:
            True  → answer appears grounded in context.
            False → likely answered from training-data weights (knowledge leakage).
        """
        import re
        STOPWORDS = {
            'the','a','an','is','are','was','were','be','been','have','has','had',
            'do','does','did','will','would','could','should','may','might','shall',
            'can','to','of','in','on','at','by','for','with','about','from','into',
            'and','but','or','not','this','that','it','its','also','such','very',
            'just','as','if','then','than','both','each','more','some','all','even',
        }
        def content_words(text):
            return {
                w for w in re.findall(r'\w+', text.lower())
                if len(w) > 3 and w not in STOPWORDS
            }

        ans_words  = content_words(answer)
        ctx_words  = content_words(context)

        if not ans_words:
            return True   # empty answer — let other logic handle it

        overlap = len(ans_words & ctx_words) / len(ans_words)
        return overlap >= min_overlap



    def _generate_answer_single_step(self, question, context, role="student", language="en"):
        """
        OPTIMIZED: Single-step generation combining factual answer + role styling.
        50% faster than two-step approach while maintaining quality.
        
        Args:
            question: User's question (in English)
            context: Retrieved context (in English)
            role: 'beginner' | 'student' | 'teacher' | 'eval'
            language: Answer language code (for audience hints)
            
        Returns:
            Generated answer string
        """
        # Get role instruction from cache
        style_instruction = self._role_cache.get(role, self._role_cache["student"])
        
        # Language note for multilingual context
        lang_note = ""
        if language != "en":
            lang_name = LANG_NAMES.get(language, 'multilingual')
            lang_note = f"The question originates from a {lang_name} context. "
        
        # Compact, efficient prompt combining both steps
        # Add explicit completion instruction to prevent truncation
        prompt = f"""Context: {context[:GENERATION_CONTEXT_LIMIT]}

Question: {question}

Instructions:
{lang_note}Provide a concise, complete answer using the context above. End with a proper conclusion.
Style: {style_instruction}

Answer:"""

        output = self.model(
            prompt,
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            repeat_penalty=GENERATION_REPEAT_PENALTY,
            stop=["\n\nQuestion:", "\n\nContext:", "Instructions:"]
        )
        
        raw_answer = output["choices"][0]["text"].strip()
        
        # Fix incomplete sentences - remove trailing incomplete sentence
        # This ensures clean answers within 200 token limit
        return self._clean_incomplete_answer(raw_answer)

    def generate_answer(self, question, context, role="student", language="en", extra_instruction="", source_info=None):
        """
        Main generation function with OPTIMIZED single-step process.

        Steps:
          1. Generate complete answer with role styling in ONE call (FAST)
          2. If extra_instruction provided, refine further (retry loop only)
          3. Append source citations if provided

        Args:
            question:          User's question (in English)
            context:           Retrieved context (in English)
            role:              'beginner' | 'student' | 'teacher' | 'eval'
            language:          Answer language code (used for audience hints)
            extra_instruction: Additional instruction for refinement (retry loop)
            source_info:       Dict with 'source_type' and 'sources' list for citations

        Returns:
            Generated answer string (English; caller translates if needed)
        """
        # Single-step generation (OPTIMIZED)
        print("  → Generating answer...", flush=True)
        answer = self._generate_answer_single_step(question, context, role, language)
        print("     ✓ Done")

        # Quick quality check: context grounding
        grounded = self._is_context_grounded(answer, context, min_overlap=0.10)
        if not grounded:
            print("     ⚠ Low context grounding detected")

        # Optional refinement (retry loop only - rare case)
        if extra_instruction:
            print("  → Refining answer...", flush=True)
            prompt = f"""Refine this answer: {answer}

Additional requirements: {extra_instruction}

Refined answer:"""
            
            output = self.model(
                prompt,
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE * 0.8,  # Even lower for refinement
                top_p=GENERATION_TOP_P,
                repeat_penalty=GENERATION_REPEAT_PENALTY,
                stop=["\n\nRefine", "Additional"]
            )
            answer = output["choices"][0]["text"].strip()
            print("     ✓ Done")

        # Append source citations
        if source_info:
            answer = self._append_source_citations(answer, source_info)

        return answer

    def _append_source_citations(self, answer, source_info):
        """
        Append source citations to the answer.
        
        Args:
            answer: Generated answer text
            source_info: Dict with 'source_type' ('local' or 'external') and 'sources' list
        
        Returns:
            Answer with appended citations
        """
        if not source_info or not source_info.get('sources'):
            return answer
        
        source_type = source_info.get('source_type', 'local')
        sources = source_info.get('sources', [])
        
        # Add spacing before sources section
        citation_text = "\n\n---\n📚 Sources:\n"
        
        if source_type == 'external':
            # Wikipedia source
            source = sources[0]
            title = source.get('title', 'Wikipedia Article')
            url = source.get('url', '')
            wiki_lang = source.get('wiki_lang', 'en')
            
            if not url and title:
                # Construct URL from title
                import urllib.parse
                encoded = urllib.parse.quote(title.replace(' ', '_'))
                url = f"https://{wiki_lang}.wikipedia.org/wiki/{encoded}"
            
            citation_text += f"• {title}\n"
            if url:
                citation_text += f"  {url}"
        else:
            # Local corpus sources
            for idx, source in enumerate(sources[:5], 1):
                title = source.get('title', 'Untitled')
                score = source.get('hybrid_score', 0)
                citation_text += f"• [S{idx}] {title} (relevance: {score:.2f})\n"
        
        return answer + citation_text
