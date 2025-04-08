import requests
# Доброе утро ;)  Код простой, но чтобы сделать его простым и коротким, а также настроить модель, пришлось немного потупить и посидеть, сори!)
API_URL = "http://localhost:4891/v1/chat/completions"

# эта штука читает файл инпут и берет его содержимое как то, с чем ллм работать
with open("input_prompt.txt", "r", encoding="utf-8") as file:
    prompt_text = file.read()

# вот тут у меня записан постоянный промт который дает задачу ллм что делать независимо от содержимого в ткст файле
constant_instruction = (
    "Your task is to summarize the given text without losing the structure and sense of it. "
    "Make sure to capture all the key points and maintain the overall logical flow."
)
# тут просто вызов ллм и ограничение модели по токенам - я понял, что все-таки можно больше 2000 сори! А также параметр temperature отвечает за креативность модели в подборе ответа - если будет нужно, объясню в тг
payload = {
    "model": "Llama 3 8B Instruct",
    "messages": [
        # This system message remains constant for every request.
        {"role": "system", "content": constant_instruction},
        # The text read from the file is provided as the user message.
        {"role": "user", "content": prompt_text}
    ],
    "temperature": 0.7,
    "max_tokens": 15000
}
# здесь код просто извлекает готовый сжатый текст из модели и переносит его в папку с llm_response.txt
headers = {"Content-Type": "application/json"}

response = requests.post(API_URL, headers=headers, json=payload)
result = response.json()
generated_text = result['choices'][0]['message']['content']

# Save the generated text into a file
with open("llm_response.txt", "w", encoding="utf-8") as file:
    file.write(generated_text.strip())
