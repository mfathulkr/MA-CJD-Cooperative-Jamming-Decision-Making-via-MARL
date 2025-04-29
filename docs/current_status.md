# Proje Durum Raporu: MA-CJD (Multi-Agent Cooperative Jamming Decision)

**Tarih:** 29 Nisan 2025 (Tahmini, loglara göre)

## 1. Giriş

Bu rapor, MA-CJD projesinin mevcut uygulama durumunu, `docs` klasöründeki belirtim belgeleri (`implementation.md`, `impadd.md`, `design_document.md`, `overview.md`) ile karşılaştırarak özetlemektedir. Amaç, mevcut yetenekleri, eksiklikleri ve sonraki adımlar için potansiyel alanları belirlemektir.

## 2. Mevcut Senaryo Yapılandırması

*   **İstenen:** Kullanıcı tarafından 2 radar ve 2 jammer'lık bir senaryo belirtilmiştir.
*   **Mevcut:** `config/default.yaml` ve `config/simulation_config.yaml` dosyaları şu anda **4 radar ve 4 jammer**'lık bir senaryo için yapılandırılmıştır. Eğer 2x2 senaryo hedefleniyorsa, bu konfigürasyon dosyalarının güncellenmesi gerekmektedir.

## 3. Varlık Hareketi

*   **Radarlar:** `simulation/environment.py` içindeki `reset` fonksiyonu, radarların pozisyonlarını belgelerde belirtildiği gibi rastgele (Normal dağılımlı mesafe, Uniform dağılımlı açı) atamaktadır. Ancak, `step` fonksiyonu içinde veya `core/radar.py` içinde radar pozisyonlarını güncelleyen bir mekanizma **görünmemektedir**. Radarların `SEARCH` durumunda anten yönünü (`theta_a`) güncellediği (`update_beam_direction`) görülmektedir, ancak fiziksel olarak yer değiştirmiyorlar.
*   **Jammer'lar:** Jammer pozisyonları `config/simulation_config.yaml` dosyasından yüklenir ve `reset` veya `step` sırasında **değişmez**.
*   **Sonuç:** Mevcut durumda, simülasyon başlatıldıktan sonra varlıklar (radarlar ve jammer'lar) **statiktir**. Belgelerdeki dinamik ortam tanımıyla bu durum tam olarak örtüşmemektedir, ancak QMix gibi algoritmalar statik senaryolarda da eğitilebilir.

## 4. Ödül Hesaplama Durumu

Ödül fonksiyonu `r = r_d + r_p + r_j` olarak `simulation/environment.py` içinde hesaplanmaktadır.

*   **`r_p` (Resource Consumption Penalty):** Tam olarak belgelerdeki formüle (`r_p = –r_{p,max} – (r_{p,max} – r_{p,min}) (P_j – P_{j,min}) / (P_{j,max} – P_{j,min})`) uygun şekilde **uygulanmıştır**. `rp_min` ve `rp_max` değerleri konfigürasyon dosyasından alınmaktadır. TensorBoard grafikleri (`Rewards/r_p_avg`) beklendiği gibi zamanla daha negatif değerlere yöneldiğini (artan güç kullanımına paralel olarak) göstermektedir.
*   **`r_d` (Tracking Penalty):** Bir radar `TRACK` durumuna geçtiğinde negatif bir ödül uygulanmaktadır. Ancak belgelerde belirtilen "tehdit seviyesine göre cezanın ölçeklenmesi" kısmı **tam olarak uygulanmamış** veya basitleştirilmiştir. Mevcut kod (`environment.py`, ~satır 520 civarı), `threat_level`'ı `rd_min` ve `rd_max` arasında doğrusal bir interpolasyon yapmak için kullanmaya çalışır, ancak `threat_level`'ın kendisi `Radar` sınıfından geliyor ve nasıl tanımlandığı/güncellendiği net değildir (`Radar` sınıfında `threat_level` yerine `rd_penalty` kullanılıyor gibi görünüyor). TensorBoard grafikleri (`Rewards/r_d_avg`) bu ödülün sıfır olmadığını, -0.2 civarında dalgalandığını gösteriyor, bu da bazı radarların zaman zaman (belki de yanlış hedeflere?) kilitlendiğini ancak bunun beklenen ölçüde olmadığını düşündürebilir. *Not: Konsol loglarında r_d=0.000 görünmesi ile TensorBoard'daki negatif değerler arasında bir tutarsızlık olabilir, loglamanın nasıl yapıldığına bağlı.*
*   **`r_j` (Jamming Success Probability Reward):** Bu ödülün hesaplanması **kısmen uygulanmıştır** ve belgelerdeki karmaşık formüllere kıyasla **basitleştirmeler** içermektedir.
    *   Bastırma (`rj_suppression = pd_no_jamming - pd_true_target`) ve Aldatma (`rj_deception = 1.0 - prod_term_rj_deception`) için ayrı ayrı hesaplamalar yapılır. Bu hesaplamalar, `detection_probability` fonksiyonuna dayanır.
    *   İki tür `r_j`'nin nasıl birleştirileceği konusunda kod içinde yorumlar (`# Max might be complex interaction...`, `# Prioritize deception if present?`) bulunmaktadır, bu da nihai birleştirme stratejisinin belgelerdeki kadar net olmadığını gösterir. Mevcut kod, aldatma varsa onu, yoksa bastırmayı kullanıyor gibi görünmektedir.
    *   Aldatma ödülü için gereken `pr_ij` hesaplamasında kullanılan `w` ve `JNR0.5` parametreleri belgelerde belirtilmiş ancak kodda bulunmuyor; `_calculate_jamming_success_reward` fonksiyonunda (ki bu fonksiyon `step` içinde çağrılmıyor gibi görünüyor, `step` kendi `r_j` hesaplamasını yapıyor) placeholder değerler kullanılmış. `step` içindeki `r_j` hesaplaması daha doğrudan `Pd` değerlerine dayanıyor.
    *   TensorBoard grafikleri (`Rewards/r_j_avg`) bu ödülün arttığını göstermektedir, bu da ajanların jamming yapmayı öğrendiğini işaret eder.

## 5. Çoklu Ajan Yeteneği (Multi-Agent Capability)

*   Sistem, çoklu ajanlar (jammer'lar) için **tasarlanmış ve uygulanmıştır**.
*   **QMix:** `QMixLearner` ve `QMixer` ağları, bireysel ajan Q-değerlerini alıp merkezi bir `Q_tot` değeri üreterek ajanlar arası koordinasyonu sağlamak üzere uygulanmıştır.
*   **MAC:** `BasicMAC`, birden fazla ajandan gözlem alıp her biri için eylem üretmektedir.
*   **Environment:** Ortam, `num_jammers` ve `num_radars` parametrelerine göre yapılandırılır ve `step` fonksiyonu tüm ajanlar için bir eylem listesi bekler. Durum ve gözlem vektörleri tüm ajan/radar bilgilerini içerir.
*   **Sonuç:** Kod tabanı, çoklu ajan etkileşimini ve QMix tabanlı merkezi eğitimi desteklemektedir.

## 6. Uygulanan ve Eksik Özellikler (Gap Analysis)

Aşağıda, belgelerde (`implementation.md`, `impadd.md`) belirtilen temel özellikler ve mevcut durumları listelenmiştir:

*   **[✓] Markov Game Modeli:** Ortam (`environment.py`) durum (S), eylem (A_i), ödül (r) tanımlarını içerir.
*   **[✓] Parameterized Action Space (T_i, P_i):** Ortam ve MAC, ayrık hedef/tip (T_i) ve sürekli güç (P_i) eylemlerini destekler.
*   **[✓] QMix Algoritması:** `QMixLearner`, `QMixer` ve hedef ağ mantığı uygulanmıştır. Monotonluk kısıtlaması (`abs()`) `QMixer` içinde mevcuttur.
*   **[✓] MP-DQN Agent Mimarisi:** `RNNAgent`, belgelerde tarif edilen Aktör (güç üretimi), RNN tabanı (GRU) ve Q-başlığı (eylem değeri) yapısını içerir.
*   **[✓] Double DQN Mekanizması:** `QMixLearner` içindeki hedef Q değeri hesaplamasında uygulanmıştır (eylem seçimi için evaluation network, değerleme için target network kullanılır).
*   **[✓] Merkezi Eğitim, Merkezi Olmayan Yürütme (CTDE):** QMix doğası gereği bunu destekler. Eğitim `Q_tot` üzerinden yapılır, yürütme bireysel ajan Q-değerlerine dayanır (epsilon-greedy ile).
*   **[~] Detaylı Fizik Simülasyonu:**
    *   Radar denklemi (`calculate_echo_power`) ve Friis iletimi (`_calculate_power_at_radar`) temel formülleri uygulanmıştır.
    *   Albersheim yaklaşımı (`detection_probability`) muhtemelen `Radar` sınıfında uygulanmıştır (kodunu görmedik).
    *   Anten kazancı (`G_rj`) basitleştirilmiş olabilir (açıya bağımlılık?).
    *   Radar durum geçişleri (Search/Track) uygulanmış ancak `Confirmation` durumu eksik olabilir.
    *   Varlık hareketi **eksiktir**.
*   **[~] Ödül Fonksiyonu Detayları:**
    *   `r_p` tam uygulanmıştır.
    *   `r_d` tehdit seviyesi ölçeklemesi tam net değildir/eksiktir.
    *   `r_j` basitleştirilmiştir, özellikle aldatma kısmı ve farklı türlerin birleştirilmesi. Gerekli parametreler (`w`, `JNR0.5`) eksiktir.
*   **[✗] İstemci-Sunucu Mimarisi:** `impadd.md`'de belirtilen C++/Python istemci-sunucu yapısı mevcut kod tabanında **bulunmamaktadır**. Tüm uygulama Python içindedir.
*   **[✓] Yapılandırma Yönetimi:** YAML dosyaları (`default.yaml`, `simulation_config.yaml`) ve `main.py` içindeki yükleme/birleştirme mekanizması mevcuttur.
*   **[✓] Loglama:** TensorBoard ve konsol loglaması uygulanmıştır.

**Özetle Eksikler:** Dinamik varlık hareketi, `r_d` ve `r_j` ödüllerinin tam olarak belirtildiği gibi uygulanması, `impadd.md`'deki istemci-sunucu mimarisi.

## 7. Mevcut Eğitim Kurulumu ve Sonuçları

*   **Hiperparametreler:** `config/default.yaml` dosyasından alınır (LR=0.0005 (Critic)/0.0003 (Actor), Epsilon=0.95->0.05 (100k adımda), Gamma=0.99, Batch Size=32, Buffer Size=5000, Target Update=200 adım, Total Steps=100k).
*   **Eğitim Döngüsü:** Her 16 bölümde 4 eğitim adımı atılır (`main.py`).
*   **Cihaz:** Son düzeltmelerle **GPU (cuda)** üzerinde çalışmaktadır.
*   **TensorBoard Sonuçları (İlk ~20k adım için):**
    *   **Öğrenme İşaretleri:** Ortalama getiri (`Perf/Avg_Return`) ve adım başına ödül (`Perf/Avg_Step_Reward`) belirgin bir artış eğilimindedir. Kayıp (`Loss/train_avg`) azalmaktadır. Jamming başarı ödülü (`Rewards/r_j_avg`) artmaktadır. Bunlar, ajanların temel görevi (jamming yaparak ödülü maksimize etme) öğrendiğini gösterir.
    *   **Kaynak Kullanımı:** Güç tüketimi cezası (`Rewards/r_p_avg`) daha negatif hale gelmekte ve ortalama güç (`Perf/Avg_Power`) artmaktadır. Bu, ajanların daha etkili jamming için daha fazla güç kullanmayı öğrendiğini gösterir.
    *   **Q-Değerleri ve Gradyanlar:** Q-değerleri (`QValues/eval_qtot_avg`, `QValues/target_qtot_avg`) oldukça hızlı artmaktadır. Bu, öğrenmenin bir işareti olsa da, potansiyel bir aşırı tahmin (overestimation) sorununa veya ödül ölçeklemesine işaret edebilir. Gradyan normu (`Stats/grad_norm`) başlangıçta artıp sonra ~30 civarında sabitlenmektedir (muhtemelen `grad_norm_clip=10.0` ile ilgili?), bu da patlayan gradyan sorunu olmadığını gösterir.
    *   **Episode Uzunluğu:** `Perf/Avg_Length` grafiği beklendiği gibi sabit 100'de kalmaktadır.
    *   **Eylem Dağılımı (`ActionDist`):** Grafikler, ajanların zamanla belirli eylemleri (belirli radar/tip kombinasyonları) tercih etmeye başladığını göstermektedir. Örneğin, `ActionDist/Action_2` ve `ActionDist/Action_4`'ün popülaritesi artmaktadır. Bu, bir politika öğrenildiğini gösterir.
    *   **`r_d` Tutarsızlığı:** `Rewards/r_d_avg` grafiği sıfır değil, -0.2 civarında dalgalanıyor. Bu, bazı istenmeyen radar kilitlemelerinin hala gerçekleştiğini veya `r_d` hesaplamasında/loglamasında bir sorun olduğunu gösterebilir.

## 8. Sonuç ve Sonraki Adımlar

Mevcut kod tabanı, MA-CJD algoritmasının temel yapısını (QMix, MP-DQN, Double DQN) başarılı bir şekilde uygulamaktadır ve çoklu ajan senaryosunda öğrenme yeteneği göstermektedir (artan getiri, öğrenilen eylem tercihleri). GPU kullanımı düzeltilmiştir.

Ancak, belgelerde belirtilenlerle arasında önemli farklar bulunmaktadır:

1.  **Statik Ortam:** Varlıkların hareket etmemesi, simülasyonun dinamizmini azaltır.
2.  **Basitleştirilmiş Ödüller:** Özellikle `r_d` ve `r_j`'nin tam olarak uygulanmaması, öğrenilen politikanın hedeflenen davranıştan sapmasına neden olabilir.
3.  **Yapılandırma Farkı:** Mevcut 4x4 senaryo, istenen 2x2'den farklıdır.

**Önerilen Sonraki Adımlar:**

1.  **Senaryo Güncelleme:** `config/default.yaml` ve `config/simulation_config.yaml` dosyalarını 2 radar ve 2 jammer kullanacak şekilde güncelleyin.
2.  **Ödül Fonksiyonunu İyileştirme:**
    *   `r_d` için tehdit seviyesi ölçeklemesini belgelerdeki gibi tam olarak uygulayın (veya `Radar` sınıfındaki `rd_penalty`'nin nasıl çalıştığını netleştirin).
    *   `r_j` hesaplamasını, özellikle aldatma kısmını ve farklı türlerin birleşimini belgelerle daha uyumlu hale getirin. Gerekli parametreleri (`w`, `JNR0.5`) ekleyin ve `step` fonksiyonunun bu detaylı hesaplamayı kullandığından emin olun.
3.  **(Opsiyonel/İleri Seviye) Dinamik Ortam:** Radarların ve/veya hedefin `step` fonksiyonu içinde hareket etmesini sağlayacak mekanizmaları ekleyin. Bu, daha gerçekçi ve zorlu bir öğrenme problemi yaratacaktır.
4.  **Detaylı Analiz:** `r_d` ödülünün neden sıfır olmadığını (TensorBoard vs konsol logları) ve Q-değerlerinin neden bu kadar hızlı arttığını daha detaylı araştırın. 