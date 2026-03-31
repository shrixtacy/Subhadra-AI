"""
Build multilingual SFT instruction dataset covering:
  - Indian mythology (Ramayana, Mahabharata, Puranas, Panchatantra)
  - Indian folktales (regional stories, Jataka tales, Panchatantra)
  - General Q&A in English, Hindi, and Odia

Output: data/multilingual_sft.jsonl
Run:    python data/build_multilingual_sft.py
"""

from __future__ import annotations
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# English — Indian Mythology & Folktales
# ---------------------------------------------------------------------------

EN_MYTHOLOGY: list[tuple[str, str]] = [
    # Ramayana
    ("Who is Rama in Hindu mythology?",
     "Rama is the seventh avatar of Lord Vishnu and the hero of the epic Ramayana. He is revered as the ideal man and king, known for his righteousness, devotion to duty, and unwavering adherence to dharma."),
    ("What is the Ramayana about?",
     "The Ramayana is an ancient Sanskrit epic that narrates the life of Prince Rama of Ayodhya. It tells the story of his exile, the abduction of his wife Sita by the demon king Ravana, and his eventual victory over Ravana with the help of the monkey god Hanuman and an army of vanaras."),
    ("Who is Hanuman?",
     "Hanuman is a divine monkey warrior and devoted follower of Lord Rama. He is celebrated for his immense strength, intelligence, and unwavering devotion. He played a crucial role in the Ramayana by finding Sita in Lanka and helping Rama defeat Ravana."),
    ("Who is Ravana?",
     "Ravana is the ten-headed demon king of Lanka in the Ramayana. Despite being a great scholar and devotee of Lord Shiva, his arrogance and lust led to his downfall when he abducted Sita, leading to his defeat by Rama."),
    ("What is the significance of Diwali?",
     "Diwali, the festival of lights, celebrates the return of Lord Rama to Ayodhya after 14 years of exile and his victory over Ravana. People light lamps to symbolize the triumph of light over darkness and good over evil."),
    ("Who are the Pandavas?",
     "The Pandavas are five brothers — Yudhishthira, Bhima, Arjuna, Nakula, and Sahadeva — who are the heroes of the Mahabharata. They are the sons of King Pandu and are known for their righteousness and valor."),
    ("What is the Mahabharata?",
     "The Mahabharata is one of the two major Sanskrit epics of ancient India. It narrates the Kurukshetra War between the Pandavas and the Kauravas, and contains the Bhagavad Gita, philosophical teachings given by Lord Krishna to Arjuna on the battlefield."),
    ("Who is Krishna in Hindu mythology?",
     "Krishna is the eighth avatar of Lord Vishnu and one of the most beloved deities in Hinduism. He is known for his divine playfulness as a child, his role as a charioteer and guide in the Mahabharata, and his teachings in the Bhagavad Gita."),
    ("What is the Bhagavad Gita?",
     "The Bhagavad Gita is a 700-verse Hindu scripture that is part of the Mahabharata. It is a dialogue between Prince Arjuna and Lord Krishna on the battlefield of Kurukshetra, covering topics like duty, righteousness, devotion, and the nature of the soul."),
    ("Who is Goddess Durga?",
     "Durga is a major Hindu goddess, the embodiment of divine feminine power (Shakti). She is depicted riding a lion and carrying weapons in her multiple arms. She is celebrated for slaying the buffalo demon Mahishasura, symbolizing the victory of good over evil."),
    ("What is the story of Ganesha's elephant head?",
     "According to Hindu mythology, Goddess Parvati created Ganesha from clay and breathed life into him. When Lord Shiva returned and found a stranger guarding Parvati's chambers, he cut off the boy's head. To console Parvati, Shiva replaced the head with that of an elephant, making Ganesha the god of beginnings and remover of obstacles."),
    ("Who is Saraswati?",
     "Saraswati is the Hindu goddess of knowledge, wisdom, arts, and learning. She is depicted in white, holding a veena (musical instrument), a book, and a rosary. She is worshipped by students and artists seeking her blessings."),
    ("What is the Panchatantra?",
     "The Panchatantra is an ancient Indian collection of interrelated animal fables in Sanskrit. Composed around 3rd century BCE, it uses stories of animals to teach principles of statecraft, human conduct, and practical wisdom. Famous stories include the crow and the snake, and the lion and the rabbit."),
    ("Tell me a Panchatantra story.",
     "The Lion and the Rabbit: A fierce lion terrorized the forest, killing animals daily. The animals agreed to send one animal each day as food. One day, a clever rabbit was chosen. He arrived late and told the lion that another lion had delayed him. The furious lion demanded to see this rival. The rabbit led him to a well and showed him his own reflection. The lion roared and jumped in, drowning himself. The clever rabbit saved the forest through wit, not strength."),
    ("Who is Lord Shiva?",
     "Lord Shiva is one of the principal deities of Hinduism, part of the Holy Trinity (Trimurti) along with Brahma and Vishnu. He is the destroyer and transformer, associated with meditation, yoga, and the arts. He is depicted with a crescent moon, the Ganges river flowing from his hair, a trident, and a serpent around his neck."),
    ("What is the story of Savitri and Satyavan?",
     "Savitri was a princess who chose to marry Satyavan, a prince destined to die within a year. When Yama, the god of death, came to take Satyavan's soul, Savitri followed him. Impressed by her devotion and wisdom, Yama granted her boons. She cleverly asked for her husband's life without directly asking for it, and Yama, bound by his word, restored Satyavan to life. This story celebrates the power of a devoted wife's love and wisdom."),
    ("What are Jataka tales?",
     "Jataka tales are a collection of over 500 stories about the previous lives of Gautama Buddha. In these stories, the future Buddha appears as various animals and humans, demonstrating virtues like compassion, wisdom, and self-sacrifice. They are an important part of Buddhist literature and Indian folklore."),
    ("Tell me the story of the Churning of the Ocean (Samudra Manthan).",
     "The Samudra Manthan is a major event in Hindu mythology. The gods (Devas) and demons (Asuras) churned the cosmic ocean using Mount Mandara as a churning rod and the serpent Vasuki as a rope. This produced many divine gifts including the goddess Lakshmi, the divine physician Dhanvantari, and the nectar of immortality (Amrita). It also produced the deadly poison Halahala, which Lord Shiva drank to save the universe, turning his throat blue — earning him the name Neelakantha."),
    ("Who is Goddess Lakshmi?",
     "Lakshmi is the Hindu goddess of wealth, fortune, prosperity, and beauty. She is the consort of Lord Vishnu and is depicted seated on a lotus flower, showering gold coins. She is worshipped during Diwali and is considered the bestower of material and spiritual wealth."),
    ("What is the story of Eklavya?",
     "Eklavya was a young tribal boy who wanted to learn archery from the great teacher Dronacharya. When Drona refused to teach him, Eklavya made a clay statue of Drona and practiced before it, becoming an exceptional archer. When Drona discovered this, he asked for Eklavya's right thumb as his fee (guru dakshina). Eklavya cut off his thumb without hesitation, demonstrating supreme devotion to his guru. This story is celebrated as an example of dedication and sacrifice."),

    # Folktales
    ("Tell me an Indian folktale about cleverness.",
     "Tenali Rama and the Thieves: Tenali Rama, the clever court jester of King Krishnadevaraya, once heard that thieves planned to rob his house. He dug a large pit in his garden and covered it with leaves. That night, he and his wife pretended to bury a pot of gold in the pit. The watching thieves dug all night but found nothing. In the morning, Tenali used the freshly dug soil to grow vegetables, turning the thieves' greed into his garden's benefit."),
    ("Who is Birbal?",
     "Birbal was one of the nine gems (Navaratnas) in the court of Mughal Emperor Akbar. He was renowned for his wit, wisdom, and quick thinking. Many popular Indian folktales feature Akbar testing Birbal's intelligence, and Birbal always finding clever solutions to seemingly impossible problems."),
    ("Tell me an Akbar-Birbal story.",
     "Akbar once asked Birbal to bring him the most foolish person in the kingdom. Birbal brought a man who was carrying a large pot of milk on his head while riding a horse. When asked why, the man said he was taking milk to the king but didn't want to spill it, so he was holding it with both hands — which meant he couldn't hold the reins. Birbal told Akbar: 'A truly foolish person doesn't realize the danger they're in.' Akbar laughed and rewarded Birbal for his wisdom."),
    ("What is the legend of Rani Padmavati?",
     "Rani Padmavati was a legendary queen of Chittor, celebrated for her extraordinary beauty and courage. When the Sultan Alauddin Khilji besieged Chittor to capture her, she led the women of the fort in Jauhar (self-immolation) to protect their honor rather than surrender. Her story is a symbol of Rajput pride, courage, and sacrifice."),
    ("Tell me about the legend of Prithviraj Chauhan.",
     "Prithviraj Chauhan was a Rajput king of the Chahamana dynasty who ruled in the 12th century. He is celebrated in Indian folklore for his valor and his love story with Princess Sanyogita. He defeated Muhammad of Ghor in the First Battle of Tarain but was later defeated in the Second Battle of Tarain. His court poet Chand Bardai composed the Prithviraj Raso, an epic poem celebrating his life."),
]

# ---------------------------------------------------------------------------
# Hindi — Indian Mythology & Folktales
# ---------------------------------------------------------------------------

HI_MYTHOLOGY: list[tuple[str, str]] = [
    ("राम कौन हैं?",
     "राम हिंदू धर्म में भगवान विष्णु के सातवें अवतार हैं। वे रामायण महाकाव्य के नायक हैं और आदर्श पुरुष, आदर्श राजा के रूप में पूजे जाते हैं। उनका जीवन धर्म, सत्य और कर्तव्य का प्रतीक है।"),
    ("रामायण की कहानी क्या है?",
     "रामायण एक प्राचीन संस्कृत महाकाव्य है जो अयोध्या के राजकुमार राम की कहानी बताता है। इसमें उनके वनवास, उनकी पत्नी सीता के रावण द्वारा अपहरण, और हनुमान तथा वानर सेना की सहायता से रावण पर विजय की कथा है।"),
    ("हनुमान जी कौन हैं?",
     "हनुमान जी एक दिव्य वानर योद्धा और भगवान राम के परम भक्त हैं। वे अपनी अपार शक्ति, बुद्धि और भक्ति के लिए प्रसिद्ध हैं। उन्होंने लंका में सीता माता को खोजा और राम की रावण पर विजय में महत्वपूर्ण भूमिका निभाई।"),
    ("महाभारत क्या है?",
     "महाभारत भारत का एक महान संस्कृत महाकाव्य है। इसमें पांडवों और कौरवों के बीच कुरुक्षेत्र युद्ध की कथा है। इसमें भगवद्गीता भी शामिल है, जो भगवान कृष्ण द्वारा अर्जुन को दिया गया दिव्य उपदेश है।"),
    ("भगवान कृष्ण कौन हैं?",
     "भगवान कृष्ण विष्णु के आठवें अवतार हैं और हिंदू धर्म के सबसे प्रिय देवताओं में से एक हैं। वे बचपन में अपनी दिव्य लीलाओं, महाभारत में सारथी और मार्गदर्शक की भूमिका, और भगवद्गीता के उपदेशों के लिए जाने जाते हैं।"),
    ("पंचतंत्र क्या है?",
     "पंचतंत्र संस्कृत में लिखी गई पशु-कथाओं का एक प्राचीन भारतीय संग्रह है। इसमें जानवरों की कहानियों के माध्यम से राजनीति, व्यावहारिक ज्ञान और जीवन के सिद्धांत सिखाए गए हैं। यह लगभग तीसरी शताब्दी ईसा पूर्व में रचा गया था।"),
    ("एक पंचतंत्र की कहानी सुनाइए।",
     "शेर और खरगोश: एक जंगल में एक क्रूर शेर रोज जानवरों को मारता था। जानवरों ने तय किया कि वे रोज एक जानवर शेर के पास भेजेंगे। एक दिन एक चतुर खरगोश की बारी आई। वह देर से पहुंचा और बोला कि रास्ते में एक और शेर ने उसे रोका। क्रोधित शेर ने उस शेर को देखने की मांग की। खरगोश उसे एक कुएं के पास ले गया और उसे अपनी परछाईं दिखाई। शेर गुस्से में कुएं में कूद गया और डूब गया। इस तरह चतुर खरगोश ने पूरे जंगल को बचाया।"),
    ("देवी दुर्गा कौन हैं?",
     "देवी दुर्गा हिंदू धर्म की प्रमुख देवी हैं जो दिव्य शक्ति का प्रतीक हैं। वे सिंह पर सवार होती हैं और उनके कई हाथों में अस्त्र-शस्त्र होते हैं। उन्होंने महिषासुर राक्षस का वध किया था, जो बुराई पर अच्छाई की विजय का प्रतीक है।"),
    ("समुद्र मंथन की कथा क्या है?",
     "समुद्र मंथन हिंदू पुराणों की एक महत्वपूर्ण घटना है। देवताओं और असुरों ने मंदराचल पर्वत को मथनी और वासुकि नाग को रस्सी बनाकर क्षीरसागर का मंथन किया। इससे लक्ष्मी, धन्वंतरि, और अमृत सहित कई दिव्य वस्तुएं निकलीं। हलाहल विष भी निकला जिसे भगवान शिव ने पीकर सृष्टि की रक्षा की, जिससे उनका कंठ नीला हो गया और वे नीलकंठ कहलाए।"),
    ("तेनाली राम कौन थे?",
     "तेनाली राम विजयनगर साम्राज्य के राजा कृष्णदेवराय के दरबार के एक प्रसिद्ध कवि और विदूषक थे। वे अपनी असाधारण बुद्धि और हास्य के लिए प्रसिद्ध थे। उनकी चतुराई की अनेक लोककथाएं आज भी भारत में प्रचलित हैं।"),
    ("अकबर-बीरबल की एक कहानी सुनाइए।",
     "एक बार अकबर ने बीरबल से पूछा कि दुनिया में सबसे बड़ा हथियार क्या है। बीरबल ने कहा — आत्मविश्वास। अकबर को विश्वास नहीं हुआ। बीरबल ने एक दिन एक पागल हाथी को बीरबल की ओर छोड़ दिया। बीरबल ने बिना घबराए एक छोटे कुत्ते को उठाकर हाथी की ओर फेंक दिया। हाथी डरकर भाग गया। बीरबल ने कहा — यह आत्मविश्वास का ही कमाल था।"),
    ("जातक कथाएं क्या हैं?",
     "जातक कथाएं गौतम बुद्ध के पूर्व जन्मों की 500 से अधिक कहानियों का संग्रह हैं। इन कहानियों में भविष्य के बुद्ध विभिन्न पशुओं और मनुष्यों के रूप में करुणा, ज्ञान और त्याग जैसे गुणों का प्रदर्शन करते हैं। ये बौद्ध साहित्य और भारतीय लोककथाओं का महत्वपूर्ण हिस्सा हैं।"),
]

# ---------------------------------------------------------------------------
# Odia — Indian Mythology & Folktales
# ---------------------------------------------------------------------------

OR_MYTHOLOGY: list[tuple[str, str]] = [
    ("ଜଗନ୍ନାଥ ଦେବଙ୍କ କଥା କ'ଣ?",
     "ଜଗନ୍ନାଥ ଭଗବାନ ବିଷ୍ଣୁଙ୍କ ଏକ ରୂପ ଏବଂ ଓଡ଼ିଶାର ଆରାଧ୍ୟ ଦେବତା। ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିରରେ ଭଗବାନ ଜଗନ୍ନାଥ, ବଳଭଦ୍ର ଓ ସୁଭଦ୍ରା ଏକ ସାଥରେ ପୂଜିତ ହୁଅନ୍ତି। ରଥଯାତ୍ରା ଉତ୍ସବ ସମୟରେ ଲକ୍ଷ ଲକ୍ଷ ଭକ୍ତ ଏଠାରେ ଏକତ୍ରିତ ହୁଅନ୍ତି।"),
    ("ରାମାୟଣ କ'ଣ?",
     "ରାମାୟଣ ଏକ ପ୍ରାଚୀନ ସଂସ୍କୃତ ମହାକାବ୍ୟ ଯାହା ଅଯୋଧ୍ୟାର ରାଜକୁମାର ରାମଙ୍କ ଜୀବନ ବର୍ଣ୍ଣନା କରେ। ଏଥିରେ ତାଙ୍କ ବନବାସ, ସୀତାଙ୍କ ରାବଣ ଦ୍ୱାରା ଅପହରଣ ଏବଂ ହନୁମାନ ଓ ବାନର ସେନାର ସାହାଯ୍ୟରେ ରାବଣ ଉପରେ ବିଜୟ ବର୍ଣ୍ଣିତ।"),
    ("ହନୁମାନ କିଏ?",
     "ହନୁମାନ ଏକ ଦିବ୍ୟ ବାନର ଯୋଦ୍ଧା ଏବଂ ଭଗବାନ ରାମଙ୍କ ପରମ ଭକ୍ତ। ସେ ଅପ୍ରତିମ ଶକ୍ତି, ବୁଦ୍ଧି ଓ ଭକ୍ତି ପାଇଁ ପ୍ରସିଦ୍ଧ। ଲଙ୍କାରେ ସୀତାଙ୍କୁ ଖୋଜି ବାହାର କରିବା ଓ ରାମଙ୍କ ରାବଣ ଉପରେ ବିଜୟରେ ତାଙ୍କ ଭୂମିକା ଅତ୍ୟନ୍ତ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ।"),
    ("ମହାଭାରତ କ'ଣ?",
     "ମହାଭାରତ ଭାରତର ଏକ ମହାନ ସଂସ୍କୃତ ମହାକାବ୍ୟ। ଏଥିରେ ପାଣ୍ଡବ ଓ କୌରବଙ୍କ ମଧ୍ୟରେ କୁରୁକ୍ଷେତ୍ର ଯୁଦ୍ଧ ବର୍ଣ୍ଣିତ। ଭଗବଦ୍ ଗୀତା ଏହି ମହାକାବ୍ୟର ଏକ ଅଂଶ ଯେଉଁଥିରେ ଭଗବାନ କୃଷ୍ଣ ଅର୍ଜୁନଙ୍କୁ ଧର୍ମ ଓ କର୍ତ୍ତବ୍ୟ ବିଷୟରେ ଉପଦେଶ ଦେଇଛନ୍ତି।"),
    ("ପଞ୍ଚତନ୍ତ୍ର କ'ଣ?",
     "ପଞ୍ଚତନ୍ତ୍ର ସଂସ୍କୃତ ଭାଷାରେ ଲିଖିତ ପ୍ରାଚୀନ ଭାରତୀୟ ପଶୁ-କଥାର ଏକ ସଂଗ୍ରହ। ଏଥିରେ ପଶୁ-ପକ୍ଷୀଙ୍କ କଥା ମାଧ୍ୟମରେ ରାଜନୀତି, ବ୍ୟାବହାରିକ ଜ୍ଞାନ ଓ ଜୀବନ ଦର୍ଶନ ଶିଖାଯାଇଛି।"),
    ("ଓଡ଼ିଶାର ଏକ ଲୋককଥା ଶୁଣାଅ।",
     "ଏକ ଗ୍ରାମରେ ଏକ ଚତୁର ଶିଆଳ ଥିଲା। ଦିନେ ସେ ଏକ ଗଭୀର କୂଅ ଭିତରେ ପଡ଼ି ଗଲା। ଏକ ଛେଳି ଆସି ଦେଖିଲା। ଶିଆଳ ବୋଲିଲା, 'ଏ ଜଳ ଅତ୍ୟନ୍ତ ମିଠା, ଆସ ପିଅ।' ଛେଳି ଲୋଭରେ ଭିତରକୁ ଡ଼େଇଁ ପଡ଼ିଲା। ଶିଆଳ ଛେଳିର ପିଠ ଉପରେ ଚଢ଼ି ବାହାରକୁ ଡ଼େଇଁ ଗଲା। ଏ କଥା ଶିଖାଏ — ଅଜଣା ଜାଗାରେ ଯିବା ପୂର୍ବରୁ ଭଲ ଭାବରେ ଚିନ୍ତା କର।"),
    ("ଭଗବାନ ଶ୍ରୀ କୃଷ୍ଣ କିଏ?",
     "ଭଗବାନ ଶ୍ରୀ କୃଷ୍ଣ ବିଷ୍ଣୁଙ୍କ ଅଷ୍ଟମ ଅବତାର। ସେ ଶୈଶବ ଲୀଳା, ଗୋପୀ ଭକ୍ତି, ଏବଂ ଭଗବଦ୍ ଗୀତାର ଉପଦେଶ ପାଇଁ ପ୍ରସିଦ୍ଧ। ମହାଭାରତ ଯୁଦ୍ଧରେ ସେ ଅର୍ଜୁନଙ୍କ ସାରଥି ଥିଲେ।"),
    ("ଦୁର୍ଗା ଦେବୀ କିଏ?",
     "ଦୁର୍ଗା ଦେବୀ ହିନ୍ଦୁ ଧର୍ମର ଏକ ପ୍ରମୁଖ ଦେବୀ ଯିଏ ଦିବ୍ୟ ଶକ୍ତିର ପ୍ରତୀକ। ସେ ସିଂହ ଉପରେ ଆରୋହଣ କରି ଅନେକ ଅସ୍ତ୍ର ଧାରଣ କରନ୍ତି। ମହିଷାସୁର ବଧ ଦ୍ୱାରା ସେ ଅଶୁଭ ଉପରେ ଶୁଭର ବିଜୟ ଦର୍ଶାଇଛନ୍ତି।"),
]

# ---------------------------------------------------------------------------
# Assemble & write
# ---------------------------------------------------------------------------

def build_dataset() -> list[dict]:
    samples: list[dict] = []

    for q, a in EN_MYTHOLOGY:
        samples.append({"question": q, "answer": a, "lang": "en"})

    for q, a in HI_MYTHOLOGY:
        samples.append({"question": q, "answer": a, "lang": "hi"})

    for q, a in OR_MYTHOLOGY:
        samples.append({"question": q, "answer": a, "lang": "or"})

    random.shuffle(samples)
    return samples


def main() -> None:
    out_path = Path(__file__).parent / "multilingual_sft.jsonl"
    samples  = build_dataset()
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} samples → {out_path}")
    lang_counts = {}
    for s in samples:
        lang_counts[s["lang"]] = lang_counts.get(s["lang"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  [{lang}] {count} samples")


if __name__ == "__main__":
    main()
