""""
Bu test dosyasında bağlam analizi yapılmaktadır.
Eğer içerik, otel, kafe veya restoran ile ilgili bir konuya dair bilgi içeriyorsa, "bağlam içi" olarak değerlendirilir.
Aksi durumda, yani bu konularla ilgisi olmayan bir içerik tespit edildiğinde, "bağlam dışı" olarak işaretlenmektedir.

Kullanılan model, açık kaynaklı ve ücretsizdir.
Genel performansı oldukça iyi olmasına rağmen, GPT tabanlı bir modelin sunduğu performansı sağlamamaktadır.

ilk kullanımda model(facebook/bart-large-mnli) yükleneceği için kodun çalışması biraz gecikebilir.
"""


from transformers import pipeline
from deep_translator import GoogleTranslator
import fasttext


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
model = fasttext.load_model("lid.176.bin")



def classification(text):
    detected_language = model.predict(text)[0][0].replace("__label__", "")
    labels = ["Otel", "Kafe", "Restoran"]
    result = classifier(text, candidate_labels=labels)

    if result['scores'][0] > 0.7:
        answer = f"Bağlam içi, Skor: {result['scores'][0]:.2f}"
    else:
        answer = f"Bağlam Dışı, Skor: {result['scores'][0]:.2f}"


    translated_text = GoogleTranslator(source='auto', target=detected_language).translate(answer)
    return translated_text



while True:
    user_input = input("> ")
    print(classification(user_input))