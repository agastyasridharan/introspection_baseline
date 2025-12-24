import os
from typing import Optional
from openai import OpenAI
from all_prompts import coherence_prompt, thinking_about_word_prompt, affirmative_response_prompt, affirmative_response_followed_by_correct_identification_prompt, mcq_correct_prompt, injection_strength_correct_prompt
# Verify API key is set
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Run: export OPENAI_API_KEY='your-key'")

# based on the 'grader prompts' section of introspection paper (https://transformer-circuits.pub/2025/introspection/index.html)
client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment
def query_llm_judge(question = None, response = None, word = None, grading_type = None, options_text = None, correct_letter = None, expected_category = None) -> Optional[bool]:
    if grading_type == "coherence":
        prompt = coherence_prompt.format(prompt=question, response=response)
    elif grading_type == "thinking_about_word":
        prompt = thinking_about_word_prompt.format(question=question, response=response, word=word)
    elif grading_type == "affirmative_response":
        prompt = affirmative_response_prompt.format(prompt=question, response=response)
    elif grading_type == "affirmative_response_followed_by_correct_identification":
        prompt = affirmative_response_followed_by_correct_identification_prompt.format(question=question, response=response, word=word)
    elif grading_type == "mcq_correct":
        prompt = mcq_correct_prompt.format(options_text=options_text, correct_letter=correct_letter, response=response)
    elif grading_type == "injection_strength_correct":
        prompt = injection_strength_correct_prompt.format(expected_category=expected_category, response=response)
    try: 
        completion = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}]
        )
        judge_response_text = completion.choices[0].message.content
        print(judge_response_text)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    if "YES" in judge_response_text:
        return True
    elif "NO" in judge_response_text:
        return False
    else:
        print(f"Warning: Unclear judge response: {judge_response_text}")
        return None

def parse_yes_no(response_text):
    """
    Parse binary YES/NO from model response for two-stage format.
    
    Args:
        response_text: The model's response string
        
    Returns:
        "YES", "NO", or None (if unclear/ambiguous)
    """
    if not response_text:
        return None
        
    response_upper = response_text.strip().upper()
    
    # Check for clear YES or NO
    has_yes = "YES" in response_upper
    has_no = "NO" in response_upper
    
    if has_yes and not has_no:
        return "YES"
    elif has_no and not has_yes:
        return "NO"
    else:
        # Ambiguous (contains both, or neither)
        print(f"Warning: Ambiguous YES/NO response: {response_text}")
        return None


def extract_concept_match(response_text, target_concept):
    """
    Check if response contains the target concept.
    Used for stage 2 of two-stage format.
    
    Args:
        response_text: The model's response string
        target_concept: The concept name to check for (e.g., "betrayal")
        
    Returns:
        True if target_concept is found in response, False otherwise
    """
    if not response_text or not target_concept:
        return False
    
    response_lower = response_text.strip().lower()
    target_lower = target_concept.lower()
    
    # Simple substring matching
    if target_lower in response_lower:
        return True
    
    # Handle common variations (e.g., "betray" matches "betrayal")
    # Could add more sophisticated matching here if needed
    
    return False