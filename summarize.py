from transformers import AutoTokenizer, AutoModelForCausalLM


def construct_prompt(transcription: str) -> str:
    return f"""You are a clinical experts who are summarizing the conversation between a patient and a doctor.
The patient is describing their symptoms and the doctor is asking questions to understand the patient's condition.
Please note that the transcription and summarization is continuous, so you should summarize the conversation as it progresses.
We could not tell patient and physician apart, so you should summarize the conversation as a whole.
Your output language should be the same as the input language.
Here is the transcription of the conversation: {transcription}
"""


class TranscriptionSummarizer:
    def __init__(self, model_name="Qwen/Qwen3-4B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def __call__(self, transcription: str) -> str:
        prompt = construct_prompt(transcription)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()


TranscriptionSummarizer()
