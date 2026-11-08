QUOTEX AI SIGNAL LAB — FINAL PROJECT ZIP
========================================

इस ZIP में क्या है?
------------------
1) PythonEngine/trading_signal_bot.py
   - Streamlit आधारित वास्तविक market research dashboard
   - 15 real-market symbols
   - 15 OTC symbol choices (OTC provider feed मिलने पर ही)
   - EMA, SMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic,
     CCI, Williams %R, OBV, ROC आदि indicators
   - 22 candle / price-action pattern checks
   - Random Forest AI model और chronological holdout check
   - 1, 2, 5 minute horizons
   - लाल/हरी candles, chart, start/stop, auto refresh और voice alert
   - 1 minute के लिए 90% model-score gate + EMA trend alignment
   - 2 minute 82%, 5 minute 78% model-score gate + trend alignment

2) AndroidApp/
   - Android Studio project source
   - WebView-based mobile interface
   - Redmi Note 9 Pro सहित Android 6+ target (minSdk 23)
   - Real / OTC selector, 15+15 asset list, 1/2/5 minute selector,
     voice test, CALL/PUT note buttons, risk and engine information
   - अभी Android shell में कोई live feed नहीं जुड़ा है; यह fake data नहीं बनाता.

3) Original_VENICE_Q_incomplete_fragment.txt
   - आपकी पुरानी VENICE Q HTML फाइल का original copy.
   - यह fragment अधूरा था, इसलिए इसे reference के रूप में रखा है.

महत्वपूर्ण सच्चाई
-----------------
- Quotex का public official API उपलब्ध नहीं है. इस project से Quotex account
  login, broker connection या automatic trade execution नहीं होता.
- Python का REAL MARKET mode Yahoo Finance (yfinance) feed लेता है. यह delayed,
  limited या अनुपलब्ध हो सकता है; इसे Quotex executable price feed न मानें.
- OTC prices broker-specific होते हैं. OTC mode चलाने के लिए licensed provider
  का OHLCV endpoint चाहिए. बिना provider के app स्पष्ट error देगा; भाव नहीं गढ़ेगा.
- Android shell अभी UI wrapper है. Python engine अलग से चलाया जाता है.
- AI का score calibrated win probability नहीं है. 90% score भी जीत की guarantee
  नहीं है. पहले paper testing करें.
- Martingale से नुकसान तेजी से बढ़ सकता है. इसे avoid करना बेहतर है; यदि कभी
  इस्तेमाल करें तो अधिकतम 2 recovery steps से आगे न जाएँ.

PYTHON ENGINE — WINDOWS / COMPUTER
----------------------------------
1. Python 3.10 या 3.11 install करें.
2. इस folder में terminal खोलें: PythonEngine
3. चलाएँ:
       pip install -r requirements.txt
       streamlit run trading_signal_bot.py
4. Browser में खुलने वाला local dashboard इस्तेमाल करें.

ANDROID APK BUILD — ANDROID STUDIO
---------------------------------
1. Android Studio install करें (computer आवश्यक है).
2. AndroidApp folder को Android Studio में Open करें.
3. Gradle sync पूरा होने दें; Android Gradle Plugin और SDK download होंगे.
4. Build > Build Bundle(s) / APK(s) > Build APK(s) चुनें.
5. APK build होने के बाद Android Studio के notification में Locate पर tap करें.
6. APK को Redmi Note 9 Pro में भेजें, file खोलें और Android की अनुमति देकर install करें.

इस ZIP में prebuilt APK शामिल नहीं है. इस environment में Android SDK/Gradle build
चलाकर device पर test नहीं किया गया, इसलिए APK बनने का झूठा दावा नहीं किया जा रहा.

अगला development step
---------------------
Android interface को Python engine / verified live market-data backend से जोड़ना
होगा. इसके लिए secure server endpoint और data provider credentials चाहिए. API
keys को APK या source code में hard-code न करें.

फ़ोल्डर संरचना
--------------
Quotex_AI_Signal_Final/
  README_HINDI.txt
  PythonEngine/
    trading_signal_bot.py
    requirements.txt
    .env.example
  AndroidApp/
    settings.gradle
    build.gradle
    gradle.properties
    app/build.gradle
    app/src/main/AndroidManifest.xml
    app/src/main/java/com/example/quotexsignal/MainActivity.java
    app/src/main/assets/index.html
  Original_VENICE_Q_incomplete_fragment.txt
