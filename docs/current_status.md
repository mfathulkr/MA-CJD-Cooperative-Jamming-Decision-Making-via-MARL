# Proje Durum Raporu: MA-CJD (Multi-Agent Cooperative Jamming Decision)

**Tarih:** [Oto-Güncelleme Tarihi: YYYY-MM-DD]

## 1. Genel Bakış

Bu rapor, MA-CJD projesinin mevcut kod tabanının durumunu (`main.py`, `simulation/`, `core/`, `learners/`, `modules/`, `runners/` vb. dizinlerdeki kodlar incelenerek) değerlendirmektedir. Proje, QMix algoritmasını kullanarak çoklu jammer ajanlarının, çoklu radarlara karşı kooperatif bir şekilde hedef seçimi (ayrık), karıştırma modu (ayrık) ve güç seviyesi (sürekli) kararları vermesini sağlayan bir Çoklu Ajan Pekiştirmeli Öğrenme (MARL) sistemidir. MP-DQN (Multi-Pass Deep Q-Network) prensipleri ajan mimarisinde kullanılır.

## 2. Uygulama Mimarisi ve Bileşenler

*   **Çalıştırma:** `main.py` betiği, yapılandırma dosyalarını (`config/default.yaml` ve `config/simulation_config.yaml`) yükler, TensorBoard loglamasını ayarlar, `simulation.environment.ElectromagneticEnvironment` ortamını, `core.mac.BasicMAC` kontrolcüsünü, `core.qmix.QMixLearner` öğrenicisini, `utils.replay_buffer.EpisodeReplayBuffer` tamponunu ve `runners.episode_runner.EpisodeRunner` çalıştırıcısını başlatır.
*   **Yapılandırma:** Simülasyon (`simulation_config.yaml`) **2 radar ve 2 jammer** için yapılandırılmıştır. Ortam (`ElectromagneticEnvironment`) bu yapılandırmayı yükler ve ilgili varlıkları (`Radar`, `Jammer`) oluşturur.
    *   Doğrulama: `ElectromagneticEnvironment` içinde hesaplanan `state_dim` ve `action_dim_discrete` değerleri, yapılandırma ve formüllere uygun görünmektedir. (`state_dim` radar/jammer özelliklerine ve `max_radar_types`'a bağlıdır, `action_dim_discrete = 2 * num_radars + 1`).
*   **Cihaz Yönetimi:** `main.py` CUDA kullanılabilirliğini kontrol eder ve buna göre `torch.device` ayarlar. Ağlar (`MAC`, `QMixLearner` içindeki ağlar) uygun cihaza taşınır.

## 3. Varlık Hareketi

*   **Radarlar:** `simulation/environment.py` içindeki `reset` fonksiyonu, radarların pozisyonlarını `simulation_config.yaml` dosyasından okur. `step` fonksiyonu içinde radarın fiziksel pozisyonunu güncelleyen bir mekanizma **bulunmamaktadır**. Ancak, `core/radar.py` içindeki `Radar` sınıfı, `SEARCH` durumunda iken anten yönünü (`theta_a`) güncelleyebilir (bu güncelleme mantığı `environment.py` içinde `step` fonksiyonunda çağrılır).
*   **Jammer'lar:** Jammer pozisyonları `config/simulation_config.yaml` dosyasından yüklenir ve simülasyon sırasında (`reset` veya `step`) **değişmez**.
*   **Sonuç:** Mevcut durumda, simülasyon başlatıldıktan sonra varlıkların **fiziksel konumları statiktir**. Yalnızca radar anten yönü (`theta_a`) güncellenmektedir. Bu, orijinal belgelerdeki potansiyel dinamik ortam tanımından farklıdır ancak QMix gibi algoritmalar statik senaryolarda da eğitilebilir.

## 4. Ödül Hesaplama Durumu (`simulation/environment.py` -> `step`)

Ödül fonksiyonu `r = r_d + r_p + r_j` olarak `step` fonksiyonu içinde hesaplanmaktadır.

*   **`r_p` (Resource Consumption Penalty):** Belgelerdeki formüle (`r_p = –r_{p,max} – (r_{p,max} – r_{p,min}) (P_j – P_{j,min}) / (P_{j,max} – P_{j,min})`) uygun şekilde **uygulanmıştır**. Normalleştirilmiş güç `P_j` (0-1 aralığında) kullanılır. `rp_min` ve `rp_max` değerleri `simulation_config.yaml` dosyasındaki `environment_params.rewards` bölümünden alınır.
*   **`r_d` (Tracking Penalty):** Bir radar `TRACK` durumuna geçtiğinde negatif bir ödül uygulanır. Bu cezanın büyüklüğü, radarın `simulation_config.yaml`'da tanımlanan statik `threat_level` değeri kullanılarak, yine `environment_params.rewards` içindeki `rd_min` ve `rd_max` arasında **doğrusal olarak ölçeklenir**. Belgelerde bahsedilen "tehdit seviyesine göre cezanın ölçeklenmesi" bu şekilde **uygulanmıştır**, ancak `threat_level` dinamik olarak değişmez.
*   **`r_j` (Jamming Success Probability Reward):** Bu ödülün hesaplanması **uygulanmıştır** ancak belgelerde bahsedilen bazı spesifik parametreler (`w`, `JNR0.5`) yerine daha temel fiziksel hesaplamalara dayanmaktadır.
    *   Hesaplama, jammer'ın seçtiği eylem tipine (Bastırma/Aldatma) göre değişir.
    *   Temel olarak, her radar için karıştırma olmadan algılama olasılığı (`pd_no_jamming`) ile karıştırma sonrası algılama olasılığını (`pd_true_target` veya `pd_false_target`) karşılaştırır.
    *   Algılama olasılıkları (`Pd`), `core/radar.py` içindeki `detection_probability` fonksiyonu (Albersheim yaklaşımı) kullanılarak hesaplanır. Bu fonksiyon, radar denklemi (`calculate_echo_power`), jammer denklemi (`received_power`) ve radarın `pulse_compression_gain` (Ga) ve `anti_jamming_factor` (D) gibi parametrelerini içeren Sinyal-Gürültü Oranı (SNR) ve Karıştırma-Sinyal Oranı (JNR) hesaplamalarına dayanır.
    *   Bastırma (`rj_suppression = pd_no_jamming - pd_true_target`) ve Aldatma (`rj_deception = 1.0 - pd_true_target`) için ayrı hesaplamalar yapılır. Kod, aldatma eylemi seçildiyse `rj_deception`'ı, bastırma seçildiyse `rj_suppression`'ı kullanır. Belgelerdeki gibi karmaşık bir birleştirme stratejisi (örn. `max`) yerine doğrudan eylem tipine göre seçim yapılır.

## 5. Çoklu Ajan Yeteneği (Multi-Agent Capability)

*   Sistem, çoklu ajanlar (jammer'lar) için **tasarlanmış ve uygulanmıştır**.
*   **QMix:** `core.qmix.QMixLearner` ve `core.networks.QMixer` ağları, bireysel ajan Q-değerlerini (`core.networks.RNNAgent` tarafından üretilen) alıp merkezi bir `Q_tot` değeri üreterek ajanlar arası koordinasyonu sağlamak üzere uygulanmıştır. `QMixer`, durum girdisine dayalı hiper ağlar kullanarak monotonluk kısıtlamasını uygular.
*   **MAC:** `core.mac.BasicMAC`, birden fazla ajandan gözlem alır, `RNNAgent` kullanarak her biri için eylem ve gizli durum üretir. Epsilon-greedy eylem seçimi uygular.
*   **Environment:** Ortam, `num_jammers` ve `num_radars` parametrelerine göre yapılandırılır ve `step` fonksiyonu tüm ajanlar için bir eylem listesi (ayrık tip/hedef + sürekli güç) bekler. Durum ve gözlem vektörleri tüm ajan/radar bilgilerini içerir.
*   **Sonuç:** Kod tabanı, çoklu ajan etkileşimini ve QMix tabanlı merkezi eğitimi (CTDE) desteklemektedir.

## 6. Uygulanan ve Eksik Özellikler (Gap Analysis)

Aşağıda, MARL ve simülasyon bağlamında beklenen/belirtilen temel özellikler ve mevcut durumları listelenmiştir:

*   **[✓] Markov Game Modeli:** Ortam (`environment.py`) durum (S), eylem (A_i - ayrık+sürekli), ödül (r) tanımlarını içerir.
*   **[✓] Parameterized Action Space (T_i, P_i):** Ortam, MAC (`BasicMAC`), Ajan (`RNNAgent`), ve Öğrenici (`QMixLearner`), ayrık hedef/tip (T_i) ve sürekli güç (P_i) eylemlerini destekler. `RNNAgent`, MP-DQN yapısına uygun olarak sürekli parametreleri (P_i) üretir ve Q-değeri bu parametre ile birlikte hesaplanır.
*   **[✓] QMix Algoritması:** `QMixLearner`, `QMixer` (hiper ağlar ve monotonluk dahil) ve hedef ağ mantığı uygulanmıştır.
*   **[✓] MP-DQN Agent Mimarisi:** `core.networks.RNNAgent`, Aktör (sürekli parametre üretimi - `actor_forward`), RNN tabanı (GRU - `forward`) ve Q-başlığı (gizli durum, ayrık eylem, sürekli parametre girdisi alır - `get_q_value_for_action`) yapısını içerir.
*   **[✓] Double DQN Mekanizması:** `QMixLearner` içindeki hedef Q değeri hesaplamasında uygulanmıştır (eylem seçimi için evaluation network, değerleme için target network kullanılır).
*   **[✓] Merkezi Eğitim, Merkezi Olmayan Yürütme (CTDE):** QMix doğası gereği bunu destekler. Eğitim `Q_tot` üzerinden yapılır, yürütme bireysel ajan Q-değerlerine dayanır (epsilon-greedy ile).
*   **[~] Detaylı Fizik Simülasyonu:**
    *   **[✓]** Radar denklemi (`calculate_echo_power`) ve Friis iletimi (`_calculate_power_at_radar`, `jammer.received_power`) temel formülleri uygulanmıştır.
    *   **[✓]** Albersheim yaklaşımı (`detection_probability`) `core/radar.py` içinde belgelerdeki formüle uygun olarak uygulanmıştır.
    *   **[✓]** Anten kazançları (`gt`, `gr`, `gj`), kayıplar (`loss`, `latm`), radar kesit alanı (`rcs`), dalga boyu (`lambda_`), gürültü gücü (`pn`), darbe sıkıştırma kazancı (`pulse_compression_gain` - Ga), anti-jamming faktörü (`anti_jamming_factor` - D) gibi parametreler `simulation_config.yaml`'dan yüklenir ve hesaplamalarda kullanılır.
    *   **[✓]** Radar durum geçişleri (Search <-> Track) SNR/Pd'ye bağlı olarak `environment.py` içinde uygulanmıştır. `Confirmation` durumu tanımlı ama kullanılmıyor.
    *   **[✗]** Varlık hareketi **eksiktir** (sadece radar `theta_a` güncellenir).
*   **[~] Ödül Fonksiyonu Detayları:**
    *   **[✓]** `r_p` tam uygulanmıştır.
    *   **[✓]** `r_d` tehdit seviyesi ölçeklemesi uygulanmıştır (statik `threat_level` ile).
    *   **[~]** `r_j` uygulanmıştır ancak belgelerde/önceki durumda bahsedilen `w`, `JNR0.5` gibi spesifik parametreler yerine temel fizik ve Albersheim yaklaşımına dayalıdır. Aldatma/bastırma ayrımı ve birleştirmesi basitleştirilmiştir (doğrudan seçim).
*   **[✗] İstemci-Sunucu Mimarisi:** `impadd.md`'de (varsa) belirtilen C++/Python istemci-sunucu yapısı mevcut kod tabanında **bulunmamaktadır**. Tüm uygulama Python içindedir.
*   **[✓] Yapılandırma Yönetimi:** YAML dosyaları (`default.yaml`, `simulation_config.yaml`) ve `main.py`/`environment.py` içindeki yükleme/birleştirme mekanizması mevcuttur.
*   **[✓] Loglama:** TensorBoard (`torch.utils.tensorboard.SummaryWriter`) ve konsol loglaması (`main.py` içinde) uygulanmıştır. Ödül bileşenleri (`r_d`, `r_p`, `r_j`), kayıp, Q-değerleri, gradyan normu, ortalama güç, bölüm getirisi/uzunluğu gibi metrikler loglanır.

**Özetle Eksikler/Farklılıklar:** Dinamik varlık hareketi, `r_j`'nin önceki belgelere göre daha temel bir yaklaşımla uygulanması, `Confirmation` radar durumunun kullanılmaması, potansiyel istemci-sunucu mimarisinin olmaması.

## 7. Mevcut Eğitim Kurulumu ve Sonuçları (Loglara Göre)

*   **Hiperparametreler:** `config/default.yaml` dosyasından alınır (Örn: LR, Epsilon decay, Gamma, Batch Size, Buffer Size, Target Update Interval, Total Steps).
*   **Eğitim Döngüsü:** `main.py` içinde tanımlanır. `EpisodeRunner` ile bölüm verisi toplanır, `EpisodeReplayBuffer`'da saklanır. Belirli aralıklarla (`train_interval`) `QMixLearner` ile eğitim adımları atılır.
*   **Loglar:** TensorBoard logları (`logs/` altında) ve konsol çıktıları, öğrenme sürecini izlemek için kullanılır.
*   **Genel Gözlemler (Tipik Beklentiler/Önceki Loglara Göre):**
    *   **Öğrenme İşaretleri:** Ortalama getiri (`Perf/Avg_Return`) ve adım başına ödülün (`Perf/Avg_Step_Reward`) artması, Kayıp (`Loss/train_avg`) değerinin azalması, Jamming başarı ödülünün (`Rewards/r_j_avg`) artması beklenir. Bunlar ajanların görevi öğrendiğini gösterir.
    *   **Kaynak Kullanımı:** Güç tüketimi cezasının (`Rewards/r_p_avg`) daha negatif hale gelmesi ve ortalama gücün (`Perf/Avg_Power`) artması, ajanların daha etkili jamming için güç kullanmayı öğrendiğini gösterebilir.
    *   **Q-Değerleri:** Q-değerlerinin (`QValues/eval_qtot_avg`, `QValues/target_qtot_avg`) artması öğrenme işaretidir, ancak aşırı hızlı artışlar potansiyel aşırı tahmine (overestimation) işaret edebilir.
    *   **Gradyanlar:** Gradyan normunun (`Stats/grad_norm`) kontrol altında olması (örn. `grad_norm_clip` ile sınırlandırılmış) patlayan gradyan sorunları olmadığını gösterir.
    *   **Eylem Dağılımı (`ActionDist`):** Ajanların zamanla belirli eylem türlerini (örn. belirli radar/tip kombinasyonları) daha sık seçmeye başlaması, bir politikanın öğrenildiğini gösterir.
    *   **`r_d` Değeri:** `Rewards/r_d_avg` değerinin sıfırdan farklı (negatif) olması, bazı radarların zaman zaman `TRACK` durumuna geçtiğini gösterir. Değerin zamanla sıfıra yaklaşması (daha az takip), öğrenmenin bir işareti olabilir.

## 8. Sonuç ve Sonraki Adımlar

Mevcut kod tabanı, MA-CJD için QMix ve MP-DQN tabanlı bir MARL çözümünün temel yapısını **başarılı bir şekilde uygulamaktadır**. Çoklu ajan koordinasyonu, parametreli eylem alanı, temel fizik simülasyonu (statik ortamda), ödül fonksiyonları ve öğrenme mekanizmaları mevcuttur. Yapılandırma ve loglama altyapısı kurulmuştur.

Ana farklılıklar/eksiklikler şunlardır:

1.  **Statik Ortam:** Belgelerde ima edilen veya daha gerçekçi senaryolar için gerekli olan dinamik varlık hareketi eksiktir.
2.  **`r_j` Uygulaması:** `r_j` ödülü, temel fiziksel prensiplere ve Albersheim'a dayanarak çalışır durumdadır, ancak önceki belgelerde belirtilen daha karmaşık formülasyonlardan (örn. `w`, `JNR0.5` parametreleri ile) farklıdır. Bu, öğrenilen politikanın hedeflenen davranıştan bir miktar sapmasına neden olabilir.

**Önerilen Sonraki Adımlar:**

1.  **Doğrulama ve Test:** Mevcut statik senaryoda öğrenilen politikanın mantıklılığını ve etkinliğini daha detaylı analiz etmek. Farklı radar/jammer yapılandırmaları ile test etmek.
2.  **(Opsiyonel) Dinamik Ortam:** Proje gereksinimlerine bağlı olarak, varlık hareketi eklemek (örn. basit yörüngeler veya daha karmaşık hareket modelleri). Bu, durum/gözlem temsillerini ve potansiyel olarak ajan mimarisini etkileyebilir.
3.  **(Opsiyonel) `r_j` Geliştirmesi:** Mevcut `r_j` hesaplamasının yeterliliği değerlendirilmeli. Gerekirse, belgelerdeki daha karmaşık formülasyonları (`w`, `JNR0.5` vb. parametreleri dahil ederek) uygulamak ve etkisini gözlemlemek.
4.  **Hiperparametre Optimizasyonu:** Mevcut yapılandırma için daha iyi performans elde etmek amacıyla öğrenme oranı, epsilon decay, ağ boyutları gibi hiperparametreleri ayarlamak.
5.  **Kod İyileştirmeleri:** Kod tekrarını azaltmak, yorumları güncellemek ve potansiyel optimizasyonları (örn. vektörleştirme) uygulamak.
