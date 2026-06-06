"""
Sample UAE reviews dataset.
500 realistic bilingual reviews across 8 major UAE brands.
Mix of Arabic MSA, Gulf dialect, Arabizi, and English.
"""

import pandas as pd
import random
from datetime import datetime, timedelta

REVIEWS_RAW = [
    # ── TALABAT ──────────────────────────────────────────────────────────────
    ("الطلب وصل متأخر ساعة كاملة والأكل كان بارد تماماً. مخيب للآمال جداً", "Talabat", "arabic"),
    ("Talabat delivery in 20 mins as always. Absolutely love the tracking feature.", "Talabat", "english"),
    ("تجربة ممتازة مع طلبات عيد الفطر، سرعة التوصيل كانت مذهلة", "Talabat", "arabic"),
    ("The app crashed twice during checkout. Lost my promo code. Terrible UX.", "Talabat", "english"),
    ("وصل الأكل دافئ وكامل، شكراً طلبات", "Talabat", "arabic"),
    ("3 orders wrong this month. Customer support just sends copy-paste replies.", "Talabat", "english"),
    ("7addi zain Talabat, delivery was super fast w el akel kaan tazeej!", "Talabat", "mixed"),
    ("Driver was rude and parked far from building. Why no parking manners?", "Talabat", "english"),
    ("الخدمة تحسنت كثيراً في الأشهر الأخيرة، استمروا على هذا المستوى", "Talabat", "arabic"),
    ("Promo codes never work. App shows 20% off but charges full price every time.", "Talabat", "english"),
    ("وصل الطلب في 15 دقيقة من المطعم في دبي مارينا. فعلاً مميزين", "Talabat", "arabic"),
    ("Cold food, cold service, cold app. Everything cold except the attitude.", "Talabat", "english"),
    ("Yallah Talabat best app in UAE, ma shof mithla!", "Talabat", "mixed"),
    ("خطأ في الطلب للمرة الثالثة هذا الشهر. لن أستخدمهم مجدداً", "Talabat", "arabic"),
    ("Incredible how fast they deliver to Al Qusais. 18 minutes flat.", "Talabat", "english"),
    ("Restaurant cancelled after 45 mins. No compensation offered.", "Talabat", "english"),
    ("التطبيق سهل الاستخدام وخيارات المطاعم كثيرة جداً، أنصح به", "Talabat", "arabic"),
    ("Driver dropped food at wrong villa. Refused to go back.", "Talabat", "english"),
    ("Mashallah service, always on time for Iftar orders", "Talabat", "mixed"),
    ("الدعم الفني بطيء جداً، انتظرت 40 دقيقة على الشات", "Talabat", "arabic"),
    ("Best food app in the Middle East, hands down. Love the variety.", "Talabat", "english"),
    ("ما يجي بالوقت المحدد دايماً، كل مرة تأخير", "Talabat", "arabic"),
    ("Order tracking is very accurate. Never missed a delivery window.", "Talabat", "english"),
    ("Food was excellent but packaging was completely crushed.", "Talabat", "english"),
    ("الأسعار ارتفعت كثيراً مع رسوم التوصيل الجديدة، يصير أرخص أروح بنفسي", "Talabat", "arabic"),

    # ── EMIRATES AIRLINE ─────────────────────────────────────────────────────
    ("Business class on EK was phenomenal. Flat bed, chef meals, impeccable crew.", "Emirates", "english"),
    ("رحلتي من دبي إلى لندن كانت تجربة استثنائية، الطاقم محترف جداً", "Emirates", "arabic"),
    ("Economy seats are cramped for a 14-hour flight. Paid premium but got budget feel.", "Emirates", "english"),
    ("الترفيه على الطائرة ممتاز، شاشات كبيرة ومحتوى عربي وفير", "Emirates", "arabic"),
    ("Missed connection in DXB due to their delay. No hotel voucher offered.", "Emirates", "english"),
    ("خدمة الدرجة الأولى لا توصف، تجربة لا تُنسى من دبي إلى نيويورك", "Emirates", "arabic"),
    ("Baggage delayed 3 days to Melbourne. Compensation process is a nightmare.", "Emirates", "english"),
    ("EK flight attendants are genuinely warm, not just professional.", "Emirates", "english"),
    ("الأسعار ارتفعت بعد الجائحة لكن الجودة لا تزال تبرر السعر", "Emirates", "arabic"),
    ("Online check-in failed. Spent 90 mins at airport counter. Unacceptable.", "Emirates", "english"),
    ("حجزت رحلة للعيد وكانت ممتلئة، لكن الخدمة كانت ممتازة على كل حال", "Emirates", "arabic"),
    ("The ice cream sundae in Business Class is legendary. Small things matter.", "Emirates", "english"),
    ("تأخر الطائرة ساعتين دون أي إشعار مسبق أو اعتذار", "Emirates", "arabic"),
    ("Emirates First Class is a different planet. Worth every dirham.", "Emirates", "english"),
    ("Wi-Fi on board is outrageously expensive. AED 60 for 500MB?", "Emirates", "english"),
    ("طاقم الطائرة كان متعجرفاً بشكل غير مقبول مع المسافرين العرب", "Emirates", "arabic"),
    ("The A380 upper deck is a work of art. Emirates nailed the experience.", "Emirates", "english"),
    ("Chauffeur service to DXB included in Business. Now that's class.", "Emirates", "english"),
    ("فقدوا حقيبتي في رحلة العودة ولم يتواصلوا حتى الآن، فضيحة", "Emirates", "arabic"),
    ("Connecting in Dubai is actually pleasant with the lounge access.", "Emirates", "english"),
    ("الميل الإضافي في برنامج Skywards يصعب استخدامه، المقاعد دائماً ممتلئة", "Emirates", "arabic"),
    ("3arabizi: Emirates business class wallah ma fee mithilha, el service top top", "Emirates", "mixed"),
    ("Cabin crew forgot my meal request three times. Last Emirates flight.", "Emirates", "english"),
    ("رحلة ممتازة كما هو متوقع من طيران الإمارات، عدت بتجربة رائعة", "Emirates", "arabic"),
    ("Entertainment system froze for 4 hours. Crew couldn't reset it.", "Emirates", "english"),

    # ── CAREEM ───────────────────────────────────────────────────────────────
    ("كريم أفضل من أوبر في دبي، السائقون محترمون والأسعار معقولة", "Careem", "arabic"),
    ("Driver cancelled 5 minutes before arrival. Peak time, no alternatives.", "Careem", "english"),
    ("Super clean car, driver spoke perfect English. Will request him again.", "Careem", "english"),
    ("تطبيق كريم محتاج تحديثات كثيرة، يتعطل كثيراً في ساعات الذروة", "Careem", "arabic"),
    ("Careem GO is great value for short trips in Sharjah.", "Careem", "english"),
    ("الوصول للمطار مع كريم كان سلساً تماماً، السائق كان في الموعد بالضبط", "Careem", "arabic"),
    ("Charged double and refused to refund. Screenshots sent, no response.", "Careem", "english"),
    ("Wallah Careem khalas, el driver kaan super nice o el car was clean", "Careem", "mixed"),
    ("السائق أخذ طريقاً أطول بكثير وزاد الأجرة، لا ثقة بهذا التطبيق", "Careem", "arabic"),
    ("Fast pickup at Dubai Mall exit. Driver knew all the shortcuts.", "Careem", "english"),
    ("Surge pricing at 3x during rain. Took AED 95 for a 15-minute ride.", "Careem", "english"),
    ("كريم بريميوم يستحق فارق السعر، سيارات نظيفة ومرتاحة جداً", "Careem", "arabic"),
    ("App UI is confusing. Hard to find pickup point in malls.", "Careem", "english"),
    ("Driver rated me 1 star for no reason. Ruined my account score.", "Careem", "english"),
    ("رحلة عمل إلى شركة في جبل علي، كانت مريحة ووصلت في الوقت تماماً", "Careem", "arabic"),
    ("AC was broken in Dubai summer. Driver said it was fine. It wasn't.", "Careem", "english"),
    ("كريم للدراجات في الجميرا ممتازة، خدمة سريعة وسعر مناسب", "Careem", "arabic"),
    ("Excellent for RTA integrated journeys. Seamless metro connection.", "Careem", "english"),
    ("انتظرت 25 دقيقة والسائق قال وصل لكنه لم يكن موجوداً", "Careem", "arabic"),
    ("Careem's Quik grocery delivery is insanely fast. 10/10.", "Careem", "english"),
    ("Car smelled terrible. Driver didn't apologize even after complaint.", "Careem", "english"),
    ("التطبيق يعمل بشكل ممتاز الآن بعد التحديث الأخير، سهل وسريع", "Careem", "arabic"),
    ("Booked Careem Plus for a month. Worth it if you ride daily.", "Careem", "english"),
    ("السائق كان يكلم شخصاً على الهاتف طوال الرحلة بصوت عالٍ", "Careem", "arabic"),
    ("Preferred Careem over taxis. Consistent quality, no haggling.", "Careem", "english"),

    # ── DUBAI MALL ───────────────────────────────────────────────────────────
    ("دبي مول أكبر مركز تسوق في العالم لكن الازدحام في الأسبوع الأول رمضان لا يُحتمل", "Dubai Mall", "arabic"),
    ("The aquarium is breathtaking. Best part of any Dubai Mall visit.", "Dubai Mall", "english"),
    ("La escalera mecánica para el Dubai Fountain, vista increíble.", "Dubai Mall", "english"),
    ("Parking is a nightmare on weekends. Need better management.", "Dubai Mall", "english"),
    ("المطاعم في دبي مول راقية جداً لكن الأسعار مرتفعة جداً مقارنة بالشارقة", "Dubai Mall", "arabic"),
    ("Gold Souk section has incredible deals if you negotiate well.", "Dubai Mall", "english"),
    ("تجربة التزلج على الجليد في المول رائعة للأطفال والعائلات", "Dubai Mall", "arabic"),
    ("Fashion section rivals London and Paris. Global luxury all in one place.", "Dubai Mall", "english"),
    ("تعبنا من المشي، المول ضخم جداً بدون خريطة واضحة", "Dubai Mall", "arabic"),
    ("The Dubai Fountain show at night is one of those rare free luxuries.", "Dubai Mall", "english"),
    ("الواي فاي في المول ضعيف جداً، من المفروض يحسنونه", "Dubai Mall", "arabic"),
    ("Staff at most stores are extremely helpful and multilingual.", "Dubai Mall", "english"),
    ("ما لقيت مكان أجلس فيه بعد ساعتين من التسوق، الكراسي قليلة جداً", "Dubai Mall", "arabic"),
    ("Kinokuniya bookstore alone is worth the whole trip to Dubai Mall.", "Dubai Mall", "english"),
    ("خدمة عملاء رائعة من طاقم الأمن والموجهين، شكراً لكم", "Dubai Mall", "arabic"),
    ("Toddler facilities are excellent. Clean, spacious nursing rooms.", "Dubai Mall", "english"),
    ("الأسعار في مطاعم ذا دبي مول مرتفعة جداً مقارنة بأي مكان آخر في الإمارات", "Dubai Mall", "arabic"),
    ("Valet parking is smooth and affordable. Staff very professional.", "Dubai Mall", "english"),
    ("مشيت ساعتين ولم أجد ما أبحث عنه. التوجيه للمحلات يحتاج تحسيناً", "Dubai Mall", "arabic"),
    ("During DSF, the discounts are phenomenal. Saved over AED 2,000.", "Dubai Mall", "english"),
    ("Crowded, noisy, chaotic on Eid. But expected. Managed it well overall.", "Dubai Mall", "english"),
    ("دبي مول تجربة كاملة للعائلة، كل شيء في مكان واحد", "Dubai Mall", "arabic"),
    ("Escalators broken near the food court for two weeks. Nobody fixed them.", "Dubai Mall", "english"),
    ("3indi al mall is life, Dubai Mall aho zain!", "Dubai Mall", "mixed"),
    ("The cinema complex is world-class. Best IMAX experience in UAE.", "Dubai Mall", "english"),

    # ── BURJ AL ARAB ─────────────────────────────────────────────────────────
    ("فندق برج العرب يستحق كل ريال دفعته، تجربة حياة لن تُنسى", "Burj Al Arab", "arabic"),
    ("The butler service was just outstanding. Felt like royalty.", "Burj Al Arab", "english"),
    ("تناولنا الغداء في الفندق بدون إقامة، الأسعار خيالية لكن التجربة لا تُوصف", "Burj Al Arab", "arabic"),
    ("Afternoon tea at Sahn Eddar was immaculate. Every detail perfect.", "Burj Al Arab", "english"),
    ("الغرف أصغر مما توقعت بالنسبة لهذا السعر الفلكي", "Burj Al Arab", "arabic"),
    ("Worth every dirham for a once-in-a-lifetime stay.", "Burj Al Arab", "english"),
    ("عامل الاستقبال كان رائعاً وأرشدنا بشكل مميز وحار جداً", "Burj Al Arab", "arabic"),
    ("The helipad dinner experience is surreal. Dubai from above.", "Burj Al Arab", "english"),
    ("صالة البخور والروائح عند الدخول كانت تجربة حسية رائعة", "Burj Al Arab", "arabic"),
    ("Al Muntaha restaurant has the best sunset view in Dubai. Unmatched.", "Burj Al Arab", "english"),
    ("السباحة في المسبح الخاص للضيوف كانت من أجمل اللحظات", "Burj Al Arab", "arabic"),
    ("For the price, I expected more sq footage in the suite.", "Burj Al Arab", "english"),
    ("الخدمة على مدار الساعة ومستجيبون في أقل من دقيقتين", "Burj Al Arab", "arabic"),
    ("Private beach is pristine. Cabana service exceptional.", "Burj Al Arab", "english"),
    ("الانترنت في الغرفة كان بطيئاً بشكل غير مقبول لفندق بهذا المستوى", "Burj Al Arab", "arabic"),
    ("Check-in with a Rolls Royce is not just a gimmick — it sets the tone.", "Burj Al Arab", "english"),
    ("الإفطار في الفندق كان أفضل بوفيه تناولته في حياتي", "Burj Al Arab", "arabic"),
    ("Wallah la experience, Burj Al Arab is not a hotel it's an event", "Burj Al Arab", "mixed"),
    ("The spa deserves its own review. Six hours of absolute bliss.", "Burj Al Arab", "english"),
    ("الموظفون يتذكرون اسمك ويستخدمونه بشكل طبيعي، لمسة شخصية حقيقية", "Burj Al Arab", "arabic"),
    ("Dining experience is overpriced relative to equivalent London restaurants.", "Burj Al Arab", "english"),
    ("خيبة أمل كبيرة في خدمة الغرف ليلة رأس السنة، كانت بطيئة جداً", "Burj Al Arab", "arabic"),
    ("The gold-leaf aesthetic is a bit dated but still impressive.", "Burj Al Arab", "english"),
    ("أفضل إقامة في حياتي بكل المقاييس، سأعود بالتأكيد", "Burj Al Arab", "arabic"),
    ("Concierge arranged everything seamlessly. Restaurant, tour, helicopter.", "Burj Al Arab", "english"),

    # ── ADNOC ────────────────────────────────────────────────────────────────
    ("ADNOC fuel stations are clean and the staff speak English. Good experience.", "ADNOC", "english"),
    ("خدمة تعبئة الوقود ذاتياً في محطات أدنوك سريعة وسهلة", "ADNOC", "arabic"),
    ("ADNOC+ loyalty program is great. Points add up fast for regulars.", "ADNOC", "english"),
    ("محطة أبوظبي الجديدة نظيفة جداً والكافيه فيها ممتاز", "ADNOC", "arabic"),
    ("Queue for LPG cylinder replacement takes forever. Need better system.", "ADNOC", "english"),
    ("تطبيق ADNOC للوقود المسبق الدفع ممتاز، يوفر وقتاً كبيراً", "ADNOC", "arabic"),
    ("Car wash quality at ADNOC is surprisingly good for the price.", "ADNOC", "english"),
    ("أسعار الوقود مناسبة والمحطات في كل مكان، سهولة الوصول ممتازة", "ADNOC", "arabic"),
    ("ADNOC distribution station near ICAD was very efficient.", "ADNOC", "english"),
    ("Pump 7 was broken for two weeks at the Mussafah station. Reported, ignored.", "ADNOC", "english"),
    ("موظفو المحطة في أبوظبي الجنوب كانوا مفيدين جداً ومتعاونين", "ADNOC", "arabic"),
    ("ADNOC cafeteria food is decent and affordable for truckers.", "ADNOC", "english"),
    ("المدفوعات الرقمية في محطات أدنوك لا تعمل في بعض الأحيان", "ADNOC", "arabic"),
    ("New ADNOC stations in Abu Dhabi Al Ain Road are world-class.", "ADNOC", "english"),
    ("خدمة تبديل الزيت سريعة لكن الأسعار أعلى قليلاً من غيرها", "ADNOC", "arabic"),
    ("ADNOC+ app crashed during refueling. Had to use cash. Very inconvenient.", "ADNOC", "english"),
    ("النظافة في محطات أدنوك عموماً ممتازة وتعكس صورة احترافية", "ADNOC", "arabic"),
    ("Staff at Al Reem Island branch very courteous and fast.", "ADNOC", "english"),
    ("الانتظار في طابور تجديد الرخصة طويل جداً", "ADNOC", "arabic"),
    ("ADNOC Drive-Through is the future. No need to get out of the car.", "ADNOC", "english"),
    ("لوحة الأسعار غير واضحة ليلاً في محطة الوحدة", "ADNOC", "arabic"),
    ("Excellent EV charging infrastructure at new ADNOC stations.", "ADNOC", "english"),
    ("كاميرا الأمن في المحطة قديمة جداً، هذا لا يليق بشركة بهذا الحجم", "ADNOC", "arabic"),
    ("ADNOC 360 convenience store has great coffee. Daily stop for me.", "ADNOC", "english"),
    ("تطبيق ADNOC+ بطيء الاستجابة أحياناً ويحتاج تحديث", "ADNOC", "arabic"),

    # ── NOON ─────────────────────────────────────────────────────────────────
    ("Noon same-day delivery is actually same-day. Amazon should take notes.", "Noon", "english"),
    ("تجربة تسوق ممتازة على نون، الأسعار تنافسية والتوصيل سريع جداً", "Noon", "arabic"),
    ("Product arrived damaged, replacement took 8 days. Frustrating.", "Noon", "english"),
    ("تطبيق نون سهل الاستخدام وخيارات الدفع كثيرة ومتنوعة", "Noon", "arabic"),
    ("Flash sale items always out of stock by the time you checkout. Misleading.", "Noon", "english"),
    ("نون أحسن من أمازون للمنتجات الخليجية والعلامات التجارية المحلية", "Noon", "arabic"),
    ("Customer service resolved my issue in one call. Impressed.", "Noon", "english"),
    ("أسعار يوم النون كانت خرافية، وفرت فلوس كثيرة على اللابتوب", "Noon", "arabic"),
    ("Noon minutes grocery is unreliable. Half items missing, no apology.", "Noon", "english"),
    ("التغليف الأخضر الجديد لنون يعكس مسؤولية بيئية، أعجبني ذلك", "Noon", "arabic"),
    ("Fake reviews on the platform are a serious problem. Can't trust ratings.", "Noon", "english"),
    ("خدمة نون تجمعنا ممتازة للمنتجات الثقيلة والأجهزة الكبيرة", "Noon", "arabic"),
    ("Noon Pay cashback is legit. Got back AED 45 last month.", "Noon", "english"),
    ("وصلني منتج مختلف تماماً عما طلبته، وعملية الإرجاع أخذت أسبوعين", "Noon", "arabic"),
    ("Best UAE Black Friday deals were on Noon. Better than Dubai Mall.", "Noon", "english"),
    ("بعض البائعين على نون يضعون أسعار مبالغ بها، يجب الرقابة عليهم", "Noon", "arabic"),
    ("Noon Express sellers are consistently reliable. Others, not so much.", "Noon", "english"),
    ("التطبيق يعطل أحياناً أثناء الدفع وهذا يسبب فقدان العروض", "Noon", "arabic"),
    ("Returned three items flawlessly. Refund hit my card next day.", "Noon", "english"),
    ("نون للأعمال للشركات ممتازة، شراء بالجملة مع فواتير ضريبية سريعة", "Noon", "arabic"),
    ("Search function is terrible. Can't find specific products.", "Noon", "english"),
    ("شحن مجاني على الأوردرات فوق 200 درهم، هذا معقول جداً", "Noon", "arabic"),
    ("The app's Arabic interface is far better than most competitors.", "Noon", "english"),
    ("Seller shipped old stock. Warranty card had 2021 date.", "Noon", "english"),
    ("نون ميناء العرب الرقمي، فعلاً بناء تقني عربي يُفخر به", "Noon", "arabic"),

    # ── LULU HYPERMARKET ─────────────────────────────────────────────────────
    ("LuLu prices are unbeatable for groceries in Sharjah. Weekly staple.", "LuLu", "english"),
    ("الفريش ماركت في لولو ممتاز، خضروات وفواكه طازجة يومياً", "LuLu", "arabic"),
    ("Self-checkout machines always out of order. Staff shortage is visible.", "LuLu", "english"),
    ("الأسعار في لولو منخفضة جداً، وفرت 40% مقارنة بسبينس الأسبوع الماضي", "LuLu", "arabic"),
    ("Fish and meat section has great variety but butcher service is slow.", "LuLu", "english"),
    ("قسم الخبز الطازج في لولو مميز، خبز عربي أصيل وبأسعار معقولة", "LuLu", "arabic"),
    ("The electronics section is surprisingly well stocked and decently priced.", "LuLu", "english"),
    ("طوابير الصندوق في عطلة نهاية الأسبوع لا تطاق، تستغرق 30 دقيقة", "LuLu", "arabic"),
    ("LuLu offers the best Ramadan specials across all hypermarkets.", "LuLu", "english"),
    ("الواجهة الرقمية لتطبيق لولو محتاجة تحديث شامل", "LuLu", "arabic"),
    ("Staff helped me find imported Indian spices I couldn't locate.", "LuLu", "english"),
    ("بعض المنتجات منتهية الصلاحية وصلت لي من الموقع الإلكتروني", "LuLu", "arabic"),
    ("Parking at LuLu Hypermarket Oud Metha is impossible on weekends.", "LuLu", "english"),
    ("قسم المأكولات البحرية متنوع جداً ونظيف ورائحة الجودة واضحة", "LuLu", "arabic"),
    ("Online delivery is hit or miss. Sometimes substitutes without asking.", "LuLu", "english"),
    ("أسعار الخضروات الموسمية في لولو الأرخص في الإمارات", "LuLu", "arabic"),
    ("LuLu loyalty card saves real money. Accumulated AED 300 this Ramadan.", "LuLu", "english"),
    ("الطوابق غير منظمة وصعب إيجاد المنتجات بدون مساعدة", "LuLu", "arabic"),
    ("Indian food section in LuLu is massive. Every regional ingredient available.", "LuLu", "english"),
    ("خدمة عملاء لولو الإلكتروني بطيئة جداً في الرد", "LuLu", "arabic"),
    ("Bakery items fresh every morning. The pain au chocolat is genuinely good.", "LuLu", "english"),
    ("كثير من المنتجات منتهية الصلاحية في قسم الأدوات المنزلية", "LuLu", "arabic"),
    ("Great for bulk buying. Corporate accounts are well managed.", "LuLu", "english"),
    ("لولو هايبرماركت الشارقة الجديد أفضل بكثير من الفروع القديمة", "LuLu", "arabic"),
    ("Air conditioning in Al Barsha branch is always too cold. Unbearable.", "LuLu", "english"),
]

# Extend to 500 with variations
SENTIMENT_BOOSTERS = {
    'positive': [
        "Really impressed overall.",
        "Would highly recommend.",
        "Will definitely use again.",
        "Exceeded my expectations.",
        "Sets the benchmark for others.",
        "Genuinely one of the best in UAE.",
    ],
    'negative': [
        "Extremely disappointing.",
        "Would not recommend to anyone.",
        "Lost a loyal customer.",
        "This is unacceptable in 2025.",
        "Needs serious improvement.",
        "Management needs to address this.",
    ],
    'neutral': [
        "Average, nothing special.",
        "Meets basic expectations.",
        "Could go either way.",
        "Take it or leave it.",
        "Not bad, not great.",
    ],
}

EXTRA_ARABIC_REVIEWS = [
    ("خدمة ممتازة وأسعار مناسبة جداً، سأعود مجدداً بكل تأكيد", "general"),
    ("لم يكن ما توقعته، خيبة أمل كبيرة بصراحة", "general"),
    ("تجربة عادية لا تستحق الانتظار الطويل", "general"),
    ("خدمة رائعة من موظفين محترفين جداً ومتعاونين", "general"),
    ("أسعار مبالغ فيها جداً مقارنة بالمنافسين", "general"),
    ("سريع وموثوق، هذا كل ما نحتاجه", "general"),
    ("جودة متدنية لا تتناسب مع السعر المرتفع", "general"),
    ("تجربة ستبقى في ذاكرتي لفترة طويلة، شكراً", "general"),
    ("مشكلة في التوصيل للمرة الثانية، سأغير الخدمة", "general"),
    ("الدعم الفني استجاب بسرعة وحل المشكلة في 10 دقائق", "general"),
]


def get_sample_reviews() -> pd.DataFrame:
    """
    Return a DataFrame of 500 realistic UAE reviews.
    Columns: text, brand, language, source, timestamp
    """
    import random
    from datetime import datetime, timedelta

    random.seed(42)
    records = []

    # Core 250 reviews
    for text, brand, lang in REVIEWS_RAW:
        days_ago = random.randint(0, 180)
        dt = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
        records.append({
            'text': text,
            'brand': brand,
            'language': lang,
            'source': random.choice(['Google Reviews', 'Reddit', 'App Store', 'Twitter/X', 'TripAdvisor']),
            'timestamp': dt.isoformat(),
            'rating': _sentiment_to_rating(text),
        })

    # Extend to 500 with variations
    brands = ['Talabat', 'Emirates', 'Careem', 'Dubai Mall', 'Burj Al Arab', 'ADNOC', 'Noon', 'LuLu']
    while len(records) < 500:
        base = random.choice(REVIEWS_RAW)
        text, brand, lang = base
        if lang == 'english' and random.random() > 0.5:
            booster_cat = random.choice(list(SENTIMENT_BOOSTERS.keys()))
            text = text + ' ' + random.choice(SENTIMENT_BOOSTERS[booster_cat])
        days_ago = random.randint(0, 365)
        dt = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
        records.append({
            'text': text,
            'brand': brand,
            'language': lang,
            'source': random.choice(['Google Reviews', 'Reddit', 'App Store', 'Twitter/X', 'TripAdvisor', 'Google News']),
            'timestamp': dt.isoformat(),
            'rating': _sentiment_to_rating(text),
        })

    df = pd.DataFrame(records[:500])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    return df


def _sentiment_to_rating(text: str) -> int:
    """Heuristic star rating from text for sample data."""
    positive_kw = ['ممتاز', 'رائع', 'excellent', 'amazing', 'love', 'best', 'perfect', 'great', 'fantastic', 'wow']
    negative_kw = ['سيء', 'فظيع', 'terrible', 'worst', 'awful', 'horrible', 'never', 'disappointed', 'avoid']
    text_lower = text.lower()
    pos = sum(1 for w in positive_kw if w in text_lower)
    neg = sum(1 for w in negative_kw if w in text_lower)
    if pos > neg:
        return random.choice([4, 5])
    elif neg > pos:
        return random.choice([1, 2])
    else:
        return 3
