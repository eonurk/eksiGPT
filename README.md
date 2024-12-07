# EksiGPT

Herkese merhaba! Bu projenin amacı bir yapay bir Ekşi Sözlük kullanıcısı yaratmak.

Bunu yapmak için Generative Pre-trained Transformer (GPT) modelini hep beraber oluşturacağız yani kodunu yazacağız. Bunu da YouTube'da yapacağımız bir video serisi ile birlikte gerçekleştireceğiz.

Verisetimiz 700 adet ekşiseyler.com'dan indirdiğimiz makalelerden oluşuyor (3.4 Megabyte) yani çok küçük bir veri seti. Ama sonunda minik bir ekşici oluşturmamız için yeterli olacak :)

İlk videoda bir 2-gram modeli oluşturduk ve tahminleri karakter seviyesinde yapıyoruz. Yani bir kelimedeki sonraki karakteri tahmin ediyoruz.

Umarım beğenirsiniz!

## [Nöral Ağlarla Ekşici Kodluyoruz (EkşiGPT)](https://youtu.be/L7rsPZ1bGHw)

[![Watch the video](https://img.youtube.com/vi/L7rsPZ1bGHw/maxresdefault.jpg)](https://youtu.be/L7rsPZ1bGHw)

Tanıtım ve bigram model'in oluşturulması

Burada ilk videoyu bitirdik! Peki neler ogrendik?

- Dokümanların içerisindeki karakter sayısı boyutunu belirliyor. Bizim dokümanda 3.5 milyon karakter var yani yaklaşık 3.5 Megabyte.

- Next token prediction, yani sonraki hece tahmini en önemli fikirlerden biri. Çünkü neyi tahmin etmemiz gerektiğini artık biliyoruz.

- Normalde heceleri de tahmin edebiliriz ama biz şu anda karakterleri tahmin ediyoruz ve bunları teker teker yapıyoruz. Yani bir önceki harf diğerini takip ediyor.
  Buna literatürde bigram model deniliyor. Matematiksel olarak da şöyle:

  𝑃(𝑐𝑖∣𝑐1,𝑐2,…,𝑐𝑖−1)≈𝑃(𝑐𝑖∣𝑐𝑖−1)

  yani, bir dökümandaki i sıradaki karakteri tahmin etmek için normalde ondan önceki tüm karakterleri bilmemiz gerekir ama burada sadece i-1 inci karakteri (yani tahmin edilenden sadece bir önceki) bilmenin tüm karakterleri bilmeye eşit olduğunu varsayıyoruz. Tabi bu bi noktaya kadar doğru ve çok da iyi bir model değil! Örneğin 'anen göteğ' tahmin etmesi gibi :D

## [Dikkat Algoritmasını Kodluyoruz! (EksiGPT)](https://www.youtube.com/watch?v=u168PH3rH7A&t=818s&ab_channel=OnurKarakaslar)

[![Watch the video](https://img.youtube.com/vi/u168PH3rH7A/maxresdefault.jpg)](https://youtu.be/u168PH3rH7A)

İkinci videoda dikkat (attention) mekanizmasıni kodluyoruz. Baştan başlayıp attention nasıl çalıştığını anlatıyoruz. Daha fazla bilgi için makaleye [buradan](https://arxiv.org/abs/1706.03762) göz atabilirsiniz:

[Colab linki](https://colab.research.google.com/drive/1XuKz_puqsDnpLSAijQlMmd4N1tOHinMI?usp=sharing)

## Genel Bilgiler

Discord Kanalımız: https://discord.gg/abCgk6hMqx
