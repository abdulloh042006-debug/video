# Zootopia — “Ekran uyg‘onadi” (qisqa montaj, 1080×1920, 30 fps, 49.6 s)

Fon: sizning xonangiz (proyektor ekrani, yashil stullar, LED panellar). Personajlar: Zootopia rasmlaridan qirqib olingan
18 ta sprite, ular endi **skelet bilan jonlantirilgan**. Musiqa: sizning trekingiz (0:00–1:30 qismidan, urg‘ular bo‘yicha kesilgan).

```
out/zootopia_intro.mp4     <- tayyor qisqa video (H.264 + AAC, 49.6 s, 1080x1920, 30 fps)
python render.py --long    <- avvalgi to‘liq 90 s versiya (out/zootopia_intro_long.mp4)
assets/sprites/*.png       <- qirqilgan personajlar (shaffof PNG)  |  assets/sprites_preview.png
assets/audio/              <- track_90s.m4a (1:30 kesilgan musiqa), beats.json (beat/onset xaritasi)
```

## Qisqa montaj (`edit.py`)

Faqat eng quvnoq joylar qoldi, bo‘sh/sekin qismlar kesildi:

| chiqish vaqti | nima |
|---|---|
| 0:00–0:01.4 | **HOOK**: kelajakdan 3 ta zarbali kadr (Lion na‘rasi → sabzi proyektorga uriladi → portlash), keyin VHS “orqaga o‘rash” |
| 0:01.4–0:10 | bo‘sh xonaga sabzi dumalab keladi → Judy ekrandan sakraydi → Nick yurib keladi → Lion yurib chiqadi → hamma yig‘iladi |
| 0:10–0:16 | build-up, **katta zarba** va musiqa to‘xtab hamma muzlab qolishi (freeze) |
| 0:16–0:23 | **SABZI O‘YINI**: zanjirli pas, ikki barobar tezlikda, so‘ng proyektorga tegib hamma “!” deb qoladi |
| 0:23–0:27 | kech kelgan Flash (soat tezlashadi, salomi bit’ga to‘g‘ri keladi) |
| 0:27–0:34 | **TOSHBAQA POYGASI**: svetofor, sichqon mashinasi uchib o‘tadi, toshbaqa yutadi |
| 0:34–0:39 | Gazelle solosi → Lion na‘rasi → hamma domino bo‘lib yassilanadi |
| 0:39–0:43 | nafas → sukunat → **BOOM** |
| 0:43–0:50 | 3 urg‘uli kombo, kamera whip, katta zarba va final poza |

O‘tishlar: flash-punch, whip-pan, zoom-blur, rang-wipe. Musiqa **bit chegaralarida** kesilib 20 ms crossfade bilan
ulangan (rasm va ovoz bir xil qismlardan olinadi, shuning uchun sinxron).

## Haqiqiy jonlantirish (skelet / “puppet” animatsiya)

`engine/rigs.py` — har bir personajga suyaklar (kapsula-vaznlar), `engine/sprites.py` — rasm vaznli qatlamlarga
bo‘linib, har bir qatlam o‘z matritsasi bilan buriladi va z-tartibda birlashtiriladi (yoriqlar avtomatik bitiriladi).

* **Nick** — oyoqlarini almashtirib yuradi, qo‘li tebranadi.  **Lion** — yurganda tizzalari ko‘tariladi, mushtlari
  tebranadi, na‘ra tortganda jag‘i tushib og‘zi ochiladi (til, tishlar).
* **Toshbaqa** — 3 oyog‘i navbatma-navbat qadam tashlaydi, boshi tebranadi (poyga).
* **Flash, Clawhauser, Beaver** — qo‘l silkitadi.  **Judy** — sakrashda oyoqlari ochiladi (star-jump).
* Hammasida: nafas, quloq/dum/yubka tebranishi, ko‘z qisish, beat’ga javob.

Tashqi AI-video xizmati (Weave) bu hisobga ulanmagani uchun ishlatilmadi; hammasi mahalliy, deterministik.

## G‘oya

Xona bo‘m-bo‘sh. LED panellar musiqa urg‘usiga yonib chiqadi, ostida sabzi dumalab keladi, proyektor yoqiladi va
**filmdagi personajlar ekrandan xonaga sakrab chiqadi** (Judy, Lionheart ekrandan; Nick chapdan sirpanib; Gazelle
aylanib; Beaver shiftdan tushib; Clawhauser pastdan “photobomb”). Musiqa 20.22 s da to‘satdan jim bo‘ladi — hamma
muzlaydi (Judy havoda, Gazelle “qog‘ozdek yupqa” pirouette’da) va 20.91 da rang to‘lqini bilan qaytadi.

Trekning o‘zi uch qismdan iborat, video ham shunga quriladi:

| vaqt | musiqa | sahna |
|---|---|---|
| 0:00–0:20.2 | baland, ~139 BPM | 1–2-sahna: chiroqlar, sabzi, proyektor, kirishlar, build-up, **freeze** |
| 0:20.2–0:20.9 | to‘liq sukunat | muzlagan tableau (kulgili pauza) |
| 0:20.9–1:19.2 | sokin groove, ~130 BPM | 3-sahna “comedic chaos”: 8 ta vignette × 4 takt |
| 1:19.2–1:19.8 | sukunat | “nafas olish”: hamma cho‘kkalab, sabzi oq-qizib turadi |
| 1:19.8–1:30 | o‘rta-baland | 4–5-sahna: **portlash**, 3-urg‘uli combo, kamera whip, **final poza** (1:28.2) |

8 ta vignette (sokin qism): conga to‘lqini · sabzi pasi (zanjirli reaksiya, proyektorga urilish) · “Sloth time”
(kech kelgan Flash, kun/tun time-lapse, soat) · disco (ko‘zgu shar + EQ halqa) · photo booth (3‑2‑1, flesh,
polaroid) · poyga (toshbaqa vs sichqon mashinasi) · Gazelle solosi + Lion “roar” (domino) · build-up.

Motion graphics (matnsiz): halqalar, sparkle, starburst, halftone, god-rays, tezlik chiziqlari, konfetti/sabzi
yomg‘iri, rang-blok wipe, EQ halqa, ko‘zgu shar, soat, ter tomchilari, viewfinder, polaroid, poyga chiroqlari,
toj, dizzy yulduzlar, spotlight, YTP uslubidagi negativ miltillash.

## Personajlar: stikerdan hajmli qahramonga

Oq “stiker” konturi olib tashlandi, o‘rniga (`engine/sprites.py`, `engine/world.py`):

* **Rig (mesh deformatsiya):** nafas olish, tana egilishi, bosh qimirlashi, quloq silkinishi va pirpirashi, Nick dumi,
  Gazelle yubkasi, Lion yolsi, sabzi barglari, Flash qo‘li, toshbaqa oyoqlari, ko‘z qisish (blink). Ular
  beat’ga va o‘z tezligiga javob beradi (harakat yo‘nalishiga teskari “drag”).
* **Yorug‘lik:** xona yorug‘ligiga moslashadi, pastga qarab ambient-occlusion, uzoqdagilarga havo xiraligi,
  va sahnadagi rangli rim-light (ekran, disco, roar, build-up, final).
* **Soya va aks:** polga tushgan siluet soya + kontakt soya + plitkadagi xira aks.
* **3D:** aylanishlar haqiqiy perspektiva (yaw) bilan, tez harakatda motion smear, fondagi bust’larda DOF.
* Chekkalardagi eski fon rangi tozalandi (decontaminate), yengil sharpen + film grain.

## Musiqa

* Trek 5:54 — `0:00–1:30` olindi, oxirida 0.9 s fade-out.
* Treknning sokin qismi (0:21–1:19) boshidagidan ~20 dB past edi; kesish joyidagi **ikkala sukunat ichida** gain
  silliq oshirildi (A −1.5 dB, B +14 dB, C +9 dB), shuning uchun hech qayerda “pumping” eshitilmaydi. Istasangiz
  `tools/audio_prep.py` dagi `GAIN_DB` ni nolga qo‘ying.
* Beat xaritasi (`beats.json`) librosa bilan chiqarilgan; vizual urg‘ular o‘lchangan kechikish bo‘yicha 40 ms oldinga
  surilgan (3 ms flux bilan tekshirilgan; tayyor videoda flesh/urg‘ular audio onset’dan o‘rtacha ≈10 ms farq qiladi, 84% i ±50 ms ichida).

## Qirqish qoldiqlarini tuzatish

`tools/repair_sprites.py` (cutout → fix_sprites → repair_sprites tartibida) manba rasmlardan qayta tiklaydi:
Lion o‘ng oyog‘i (oldidagi sloth sababli kesilgan edi), Clawhauser yelkasi, Gazelle tirsagi, toshbaqa oyoqlari va
yo‘l soyasi, Judy / Nick ko‘zlari va mashina oynasi (shaffof “teshik”lar manbadan to‘ldirildi), Judy badge
yonidagi begona yengcha parchalari; hamma joyda halo (eski fon rangi) tozalandi. `tools/preview_sprites.py` — kontakt varaq.

## Qayta yaratish

```bash
pip install -r requirements.txt          # + ffmpeg
python tools/make_plate.py               # fon 1080x1920
U2NET_HOME=<isnet-general-use.onnx papkasi> python tools/cutout.py && python tools/fix_sprites.py && python tools/repair_sprites.py
python tools/audio_prep.py               # 1:30 kesish + beats.json
python render.py                         # qisqa montaj -> out/zootopia_intro.mp4  (~5 min, 4 yadro)
python render.py --long                  # to‘liq 90 s versiya
python edit.py preview /tmp/frames 0 30 400   # montaj kadrlarini ko‘rish
python tools/peek.py /tmp/frames 3.5 20.3 80.1 89.9   # alohida kadrlarni ko‘rish
```

Kod: `engine/` (sprite renderi, fon/yorug‘lik, motion graphics), `director.py` (butun xoreografiya, beat’larga
bog‘langan: `A(i)`, `B(i)`, `C(i)`), `render.py` (parallel chunk → x264 → audio mux).

> Eslatma: videoda tez miltillovchi effektlar (flesh, negativ kadrlar) bor.
> Personajlar va musiqa — o‘z egalariga tegishli; loyiha faqat shaxsiy/ijodiy foydalanish uchun.
