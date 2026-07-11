prompts = {
    "fiction": {
        "title_author": "",
        "genre": "",
        "plot_summary": {
            "premise": "1-2 sentence hook",
            "main_plot": "3-5 paragraph summary of key events",
            "climax": "The turning point",
            "resolution": "How it ends"
        },
        "characters": [
            {
                "name": "",
                "role": "protagonist/antagonist/supporting",
                "arc": "How they change throughout the story"
            }
        ],
        "themes": ["list of major themes"],
        "writing_style": "narrative voice, pacing, notable techniques",
        "emotional_impact": "intended reader experience"
    },

    "nonfiction_summary_template": {
        "title_author": "",
        "book_type": "self-help/business/academic/memoir",
        "core_argument": "Thesis statement in 1-2 sentences",
        "key_concepts": [
            {
                "concept": "",
                "explanation": "",
                "supporting_evidence": "",
                "practical_application": ""
            }
        ],
        "chapter_breakdown": [
            {
                "chapter": "",
                "main_idea": "",
                "key_takeaways": []
            }
        ],
        "methodology": "Author's approach/framework",
        "target_audience": "",
        "actionable_insights": ["Top 5-10 takeaways"]
    },

    "work_doc_summary_template": {
        "document_type": "report/proposal/meeting_notes/whitepaper",
        "purpose": "",
        "key_findings": [],
        "decisions_required": [],
        "action_items": [
            {
                "task": "",
                "owner": "",
                "deadline": ""
            }
        ],
        "risks_concerns": [],
        "next_steps": []
    }
}


CHUNK_PROMPT = """You are an expert, objective book summarizer.

First, determine the type of content in this excerpt: Fiction, Non-Fiction, or Professional/Work Document.

Then, create a clear, detailed but concise summary of the provided text excerpt.

Focus on:
- Main ideas, arguments, or plot developments
- Key details, evidence, examples, or character actions
- Important concepts, turning points, or takeaways
- Tone and style (if relevant)

Requirements:
- Be accurate and faithful to the original text
- Avoid adding external information or speculation
- Use neutral language
- Keep the summary proportional to the content length

Text excerpt:
{text}

SUMMARY:"""

SUMMARY_PROMPT = """You are an expert book summarizer synthesizing multiple section summaries into one cohesive final summary.

The input consists of summaries from different parts of the book. Combine them into a well-structured, comprehensive final summary.

First, identify the overall book type (Fiction / Non-Fiction / Work Document) and adjust the structure accordingly.

**Required Structure:**

**1. Book Overview**
- Title & Author (if known)
- One-paragraph high-level summary
- Genre/Type and core theme

**2. Key Content Breakdown**
- For Non-Fiction/Work Documents: Main arguments, chapters/sections, evidence & examples
- For Fiction: Plot arc (spoiler-free first, then detailed if requested), main characters & arcs, setting, major themes

**3. Core Insights & Takeaways**
- Most important lessons, messages, or memorable elements
- Practical applications (for non-fiction)

**4. Additional Analysis (if relevant)**
- Writing style, strengths/weaknesses, target audience

**Guidelines:**
- Maintain logical flow and coherence
- Eliminate redundancy while preserving important details
- Prioritize accuracy and balance
- Use clear headings and bullet points for readability
- Keep the tone professional yet engaging

Section summaries to synthesize:
{text}

FINAL BOOK SUMMARY:"""
