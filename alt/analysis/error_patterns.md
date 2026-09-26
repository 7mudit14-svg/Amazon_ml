# v10 stage-2 error analysis (fold-4 sample)

## Setup
- Sample: 14,942 fold-4 S1s with id < 150000 (feat_d parts 0-4), 684,841 candidate pairs, 51,805 true pairs.
- Decision reproduced: prior_shift(0.6) -> single_owner (computed over ALL S1s claiming the sample's targets, 1.6M rows) -> dta_select(missed=0). Policy from /home/user/work_v10/models/matcher_c/meta.json = {dta_own, missed 0, shift 0.6}.
- Sample macro F0.5 = 0.97617 (sealed-fold report: 0.97614). Within-sample single_owner gives 0.97617 too.
- Shift check on this (train-distribution) sample: shift 1.0 -> 0.97639, 0.8 -> 0.97588, 0.6 -> 0.97617 (the 0.6 shift is deliberate for test).
- Loss attribution: each S1's (1-F) is split equally over its error pairs (FN in candidates, FN from blocking, FP), divided by N. Total macro loss 0.02383.

## Loss by category
| category | pairs | share of macro loss |
|---|---|---|
| FN, true pair never in candidates (blocking miss) | 1656 | 49.3% |
| FN, in candidates but not predicted | 1559 | 41.3% |
| FP | 102 | 9.5% |

Stage 2 can only act on the 50.7% that is in-candidate.

## Primary pattern (exclusive, priority order), share of total macro loss
| pattern | FN_cand | FP | FN_block |
|---|---|---|---|
| blank target address, identical-name tie among >=2 claiming S1s (raw names equal) | 11.8% (482) | 0.1% | - |
| blank target address, other | 8.6% (376) | 0.5% | 9.3% (350) |
| house number differs (not truncation) | 7.0% (179) | 3.1% (24) | 2.4% |
| house number truncation (1534 vs 534, 225 vs 22) | 1.6% (44) | 0.3% | 2.4% |
| alias name (core token_set < .5), address matches | 4.3% (171) | 0.35% | 2.6% |
| Indic script target name | 2.8% (90) | 1.0% (8) | 11.1% (334) |
| typo-heavy name (core tset .5-.9) | 1.7% (77) | 0.65% | 8.8% (312) |
| legal-form variant only | 1.2% | 1.7% (17) | 1.2% |
| website-style name | 0.9% | 0.5% | 6.7% (214) |
| missing house number in target | 0.4% | 0.45% | 3.3% |
| dba/fka | ~0 | 0 | 0.1% (4 pairs) - not a real pattern in this data |
Non-exclusive tag counts are in step8_tags.py output (legal_variant tags 45% of FN_cand but mostly co-occurs with other patterns).

## What does NOT separate (tested in the p in (0.02,0.98) zone, true-rate vs mean calibrated p)
Stage-2 p is well calibrated inside every pattern; raw-string "noise signatures" carry no extra information (decoys are noised too):
accent chars in T name (on: .22 true / .22 p), digit-in-word homoglyph (.33/.28, n=231), l-for-I, duplicated word, brackets,
"#12345/(ID:)" tags, honorifics (Dr/Smt/Sri/The), double spaces, all-lower/all-upper, exact raw name (.18/.20).
Alias consistency (other candidates of the same S1 with identical target core_key; target core frequency): no lift (B: .36/.33 n=11; core_df buckets within +-0.05 of p).
House-number structure: lev distance, same length, first/last digit equal, |diff| buckets - all within ~0.05 of p except the two listed below.
Blank-slot prior (an S1 almost never has 2 blank-address targets in one source: 76 of 2141 (S1,src) with any) - fires on <1% of pairs, not usable.

## Features that do add something
1. rr_gap (blank-address targets): raw-name similarity to this S1 minus best among other S1s claiming t.
   norm(x) = rapidfuzz.utils.default_process(strip_accents(NFKD(x)))
   rr = fuzz.ratio(norm(s.business_name), norm(t.business_name))/100 over the full (s1,t) candidate table
   rr_other = max rr over other S1s claiming t (top-2 trick: mx=rr.max().over("t"); if rr<mx or count(rr==mx)>1 then mx else 2nd max)
   rr_gap = rr - rr_other ; also rt_gap with fuzz.token_sort_ratio.
   Uncertain-zone A pairs: gap>0.05 true .75 vs p .66 (n=340); (0,0.05] .52 vs .40; tie .13 vs .11; <-0.05 .054 vs .074.
   CV LightGBM on uncertain rows (logit p + 8 existing feats): logloss .1280 -> .1263, AUC .9794 -> .9801. Macro-F change on the sample is within noise (+0.0001).
2. num_prefix_trunc = s0.str.starts_with(t0) | t0.str.starts_with(s0) (first house numbers, s0 != t0); stage 2 only has ends_with (num_suffix).
   true .69 vs p .63 (n=176). |s0-t0|>100 with same length: .74 vs .64 (n=107). Small n.
3. t_addr_pobox_half = t.business_address contains PMB / PO BOX / "1/2" (pollutes addr_nums: "738 1/2" -> "738 1 2", "PMB 801" -> 801 read as the house number).
   true .70 vs .64 (n=148) and .72 vs .65 (n=58). Better fix: drop those tokens from addr_nums in normalize.

## Conclusions
- 12% of loss is identical-name ties on blank-address targets (e.g. 164 S1s named "Underwood, Stowe & Monroe Inc"): no pair feature can separate these; the expected-F rule is already right not to predict them.
- The biggest lever is blocking (49%): Indic names (11%), blank-address targets (9%), typo-heavy names (9%), website names (7%), missing/truncated house numbers (6%), alias names (3%).
- Within candidates, calibration is already tight per pattern; expected gain from new pair features is on the order of +0.0001-0.0003 macro F.

## Files
step1_predict.py (sample + decision), step2_errors.py, step3_segments.py, step4_blank.py (rr_gap; peak RSS 2.8 GB), step5_noise.py, step6_alias.py,
step7_eval.py (CV evaluation), step8_tags.py (pattern tagging), step9_nums.py. Examples: ex_fn_cand.txt, ex_fp.txt, ex_fn_block.txt (40 each).

## Examples: FN in candidates (40)
```
[30085/425928 US src2 p=0.412 pown=0.21 ntrue=4 npred=2 tp=2]
   S: 'Silver Automation Union' | '25227 Spring Iris Lane, Katy, TX'
   T: 'The Silver Automation Union' | ''
   p1=0.159 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.17 rank=4 dup=0 tn=1.79 nwnew=0
[136879/432120 India src2 p=0.726 pown=0.613 ntrue=6 npred=5 tp=5]
   S: 'Inspire & Sons Private Limited' | '12, Gayatri Vihar, Dewas, Madhya Pradesh'
   T: 'lnspire Sons Private Limited Service' | '#12, GAYATRI VIHAR, DEWAS, Madhya Pradesh'
   p1=0.346 tset=0.69 ratio=0.69 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=6 dup=0 tn=1.39 nwnew=2
[40799/509883 India src2 p=0.828 pown=0.743 ntrue=4 npred=4 tp=3]
   S: 'Marshall Durga Private Limited' | 'Mamta Devi, Ward No.04, Dudhpura Near Chaiti Durga, Samastipur, Bihar'
   T: 'Marshall Durga' | 'HN 326 MAMTA DEVI, WARD NO.04, DUDHPURA NEAR CHAITI DURGA, SAMASTIPUR, Bihar'
   p1=0.853 tset=1.00 ratio=1.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=0.69 nwnew=0
[16424/589754 US src2 p=0.806 pown=0.714 ntrue=5 npred=4 tp=4]
   S: 'Nevada Womens Health Partners PLLC' | '14528 546, Nevada, TX'
   T: 'NEVADA W0MENS HEALTH' | '546, NEVADA, TX'
   p1=0.253 tset=1.00 ratio=0.82 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.69 rank=5 dup=0 tn=1.95 nwnew=0
[63071/917052 India src2 p=0.084 pown=0.038 ntrue=6 npred=4 tp=4]
   S: 'Great Foundation' | 'Shopno2 Raghvendra Avenue, Padmanagar New Pacha Peth, Solapur North, Solapur, Maharashtra'
   T: 'Smt Great Foundation Corporation' | ''
   p1=0.013 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.01 rank=7 dup=0 tn=4.17 nwnew=0
[147518/922623 US src2 p=0.05 pown=0.022 ntrue=6 npred=3 tp=3]
   S: 'Maid Bakery! Inc' | 'IA, Des Moines, 1840 Martin Luther King Jr Parkway'
   T: 'The Maid Bakery!  Inc' | ''
   p1=0.039 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.06 rank=6 dup=0 tn=4.22 nwnew=0
[15590/988937 India src2 p=0.535 pown=0.409 ntrue=4 npred=1 tp=1]
   S: '<CITY_NAME> Homes Private Limited' | 'Ausa Road, Tq. Latur, Latur, Maharashtra'
   T: '<CITY_NAME> Homes Private' | 'NO 827 AUSA ROAD, TQ. LATUR, Maharashtra'
   p1=0.844 tset=1.00 ratio=1.00 atset=1.00 nm=0 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=2.08 nwnew=0
[25968/2165926 India src2 p=0.618 pown=0.408 ntrue=5 npred=4 tp=4]
   S: 'YU Trading Limited' | 'Rz-9C, Puran Nagar, Palam Colony, New Delhi, Delhi, South West Delhi, Delhi'
   T: 'YU Trading Ltd Ltd' | ''
   p1=0.651 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.49 rank=5 dup=0 tn=3.64 nwnew=0
[49927/2641716 US src2 p=0.368 pown=0.135 ntrue=8 npred=5 tp=5]
   S: 'F 2 S Idea' | '148 Donna Kay Drive, Greenbrier, AR'
   T: 'F 2 S Ídea' | ''
   p1=0.337 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.49 rank=8 dup=0 tn=1.39 nwnew=0
[469/3208306 India src2 p=0.72 pown=0.604 ntrue=3 npred=2 tp=2]
   S: 'Gitanjali Realestate Private Limited' | 'Khasra No 950, 950 Me And 951, Village Kilhora Jalalabad Hapur, Hapur, Ghaziabad, Uttar Pradesh'
   T: 'PRIVATEREALESTATE.COM #85158' | '950., GHAZIABAD, NEW DELHI, Uttar Pradesh'
   p1=0.828 tset=0.00 ratio=0.00 atset=0.82 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.14 rank=1 dup=0 tn=2.30 nwnew=0
[28340/3390524 India src2 p=0.636 pown=0.51 ntrue=5 npred=4 tp=4]
   S: 'Visakhapatnam Indiaprivate Pvt Ltd' | '6-55-70/4, Golla Veedhi, Tagarapuvalasa, Bheemili, Visakhapatnam, Visakhapatnam, Vishakhapatnam, Andhra Pradesh'
   T: 'ivisakhapatnam.com' | '6-55-70/4, VISAKHAPATNAM, VISHAKHAPATNAM, Andhra Pradesh'
   p1=0.984 tset=0.00 ratio=0.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.94 rank=3 dup=0 tn=4.39 nwnew=0
[143760/3812008 US src2 p=0.199 pown=0.112 ntrue=5 npred=3 tp=3]
   S: 'Bius Inc.' | '5940 2625, Roy City, UT'
   T: 'Bius Inc' | ''
   p1=0.185 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.28 rank=5 dup=0 tn=3.50 nwnew=0
[81011/4028526 US src2 p=0.167 pown=0.021 ntrue=4 npred=3 tp=3]
   S: 'Gulf Center' | '9354 Chirping Road, Hixson, TN'
   T: 'YUMAVIO' | 'HIXSON, TN, 9354  CHIRPING RD'
   p1=0.270 tset=0.11 ratio=0.11 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.76 rank=4 dup=0 tn=2.08 nwnew=1
[98929/4132917 India src2 p=0.034 pown=0.014 ntrue=5 npred=4 tp=4]
   S: 'Bright Marketing' | 'C/103, Radhe Residency, Nr. Manisha Garnala, Kosad - Utran Road, Utran, Surat, Gujarat'
   T: 'Dr BRIGHT MÁRKETING' | ''
   p1=0.015 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.02 rank=9 dup=0 tn=3.95 nwnew=0
[89779/4366706 US src2 p=0.018 pown=0.007 ntrue=7 npred=4 tp=4]
   S: 'Urban Hair Studio' | '2426 Water Street, Unit S4, Decatur, IL'
   T: 'Urban  Hair Studio & Co' | ''
   p1=0.010 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.02 rank=12 dup=0 tn=4.49 nwnew=0
[96377/5532473 US src3 p=0.237 pown=0.157 ntrue=5 npred=3 tp=3]
   S: 'Rois Hamm, D.C., M.D., P.C. LP' | '7881 Lita Corte, Rancho Cucamonga, CA'
   T: 'Rois Hamm, D.C., M.D., P.C. LLC' | '7895 Lta Corte, Rancho Cucamonga, California'
   p1=0.362 tset=1.00 ratio=1.00 atset=0.87 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=2.56 nwnew=0
[142892/5540733 US src3 p=0.382 pown=0.189 ntrue=4 npred=3 tp=3]
   S: 'Wells Frontier Open LLC' | '1200 Broadway, Unit 2214, Nashville, TN'
   T: 'wells frontier open llc' | ''
   p1=0.418 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.52 rank=6 dup=0 tn=3.09 nwnew=0
[110314/5615946 US src3 p=0.789 pown=0.679 ntrue=4 npred=3 tp=3]
   S: 'Crow Stores Inc.' | '10614 Maritca Drive, Albuquerque, NM'
   T: 'Crow Stores (Inc.)' | ''
   p1=0.610 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.30 rank=4 dup=0 tn=4.22 nwnew=0
[120156/6125303 US src3 p=0.003 pown=0.001 ntrue=5 npred=4 tp=4]
   S: 'Patriot Pediatrics LLC' | '656 Firestone Avenue, Salem, OR'
   T: 'Patriot  Pediatrics & Co' | ''
   p1=0.002 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=1 cmax=0.05 rank=6 dup=0 tn=3.09 nwnew=0
[72802/6174075 India src3 p=0.019 pown=0.007 ntrue=4 npred=3 tp=3]
   S: 'Aditya Consulting Private Limited' | 'H No-20 A, 4Th/F Okhla Village, New Delhi, South Delhi, Delhi'
   T: 'Aditya Consulting Prívate Limited' | ''
   p1=0.020 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.02 rank=9 dup=0 tn=4.01 nwnew=0
[37442/6436112 India src3 p=0.704 pown=0.505 ntrue=2 npred=1 tp=1]
   S: 'DYV Construction Private Limited' | '11/20-4, Thoppukadu, Ekkatampalayam Saanarpalayam Po, Chennimalai, Erode, Tamil Nadu'
   T: 'DYV Cónstruction Private Limited' | ''
   p1=0.620 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.46 rank=2 dup=0 tn=1.79 nwnew=0
[95010/6524044 US src3 p=0.306 pown=0.134 ntrue=6 npred=4 tp=4]
   S: 'Empire Secure Research' | 'CA, 8206 Primoak Way, Elk Grove'
   T: 'Empire Research Secure' | ''
   p1=0.250 tset=1.00 ratio=0.68 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.48 rank=6 dup=0 tn=1.39 nwnew=0
[105004/6580580 India src3 p=0.804 pown=0.637 ntrue=6 npred=5 tp=5]
   S: 'Om Impex Private Limited' | 'Geras Skyvillas Bldg No 05 Fl-V-G-Ii S.No 64 Kharadi, Pune, Maharashtra'
   T: 'ॐ Impex प्राइवेट लिमिटेड' | 'Hn 988 Geras Skyvillas Bldg No 05 Fl-v-g-ii S.no 64 Kharadi, Pune, MH'
   p1=0.619 tset=0.77 ratio=0.38 atset=0.91 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=1 dom=0 legc=0 cmax=0.26 rank=6 dup=0 tn=2.30 nwnew=3
[79592/6676887 US src3 p=0.636 pown=0.374 ntrue=4 npred=3 tp=3]
   S: 'Foot & Ankle Premier Care LLC' | '442 Hill Road, Smithville, TX'
   T: 'Foot & Ankle Premier Care Llc' | ''
   p1=0.472 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.58 rank=4 dup=0 tn=1.39 nwnew=0
[48580/6698675 US src3 p=0.386 pown=0.218 ntrue=6 npred=5 tp=5]
   S: 'Sitech Value Summit Inc.' | '130 Edgewood Place, Whitefish, MT'
   T: 'sitech value summit inc.' | ''
   p1=0.241 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.16 rank=6 dup=0 tn=3.61 nwnew=0
[20994/6902337 US src3 p=0.508 pown=0.382 ntrue=4 npred=3 tp=3]
   S: 'Kathe S. Clayton, P.A., PC' | '16 Morning Dove Drive, Westport, MA'
   T: 'PC S. Cae, P.A., Kathe' | '16 Morning Dove Dr, Westport, Massachusetts'
   p1=0.957 tset=0.83 ratio=0.44 atset=0.94 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=4 dup=0 tn=2.20 nwnew=1
[134200/6903273 US src3 p=0.049 pown=0.03 ntrue=7 npred=6 tp=6]
   S: 'West Usa' | '621 6th Avenue, Unit UNIT 5, Great Falls, MT'
   T: 'Wb Usa' | '621 6rd Avenue, N/A, Great Falls, Montana'
   p1=0.020 tset=0.71 ratio=0.71 atset=0.90 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=8 dup=0 tn=3.09 nwnew=1
[109775/7058981 India src3 p=0.454 pown=0.234 ntrue=3 npred=1 tp=1]
   S: 'First Public School' | '91 Springboard Business Hub Pvt Ltd, Gopalakrishna Complex, #45/3 Residency Road Mg Road, Bangalore North, Bangalore, Karnataka'
   T: 'firstpublicschoolcom' | '##91 Springboard Business Hub Pvt Ltd, Bangalore North, Bangalore, KA'
   p1=0.587 tset=0.69 ratio=0.69 atset=0.97 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.12 rank=3 dup=0 tn=3.85 nwnew=1
[100360/7176671 India src3 p=0.603 pown=0.427 ntrue=3 npred=2 tp=2]
   S: 'Indian Healthcare LLP' | 'A 301, Floor 3Rd, Plot 249, A, Neelam Centre Baburao Pendharkar Marg, Glaxo, Worli Co, Lony, Mumbai, Mumbai City, Maharashtra'
   T: 'इंडियन हेल्थकेयर एलएलपी' | 'A 3-01, Mumbai, MH'
   p1=0.213 tset=0.05 ratio=0.05 atset=0.80 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=1 dom=0 legc=0 cmax=0.80 rank=4 dup=0 tn=3.26 nwnew=3
[85851/7437216 US src3 p=0.62 pown=0.495 ntrue=2 npred=1 tp=1]
   S: 'Royal Hospitality Partners' | '225 Logan Street, Unit 24, Urbana, OH'
   T: 'Royal Hospitality Partners Co' | '22 Logan Street, # 24, Urbana, Ohio'
   p1=0.361 tset=1.00 ratio=1.00 atset=0.95 nm=1 nc=0 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=1 dup=1 tn=2.20 nwnew=0
[111115/8190207 US src3 p=0.189 pown=0.107 ntrue=3 npred=2 tp=2]
   S: 'Data Construction Enterprises' | '2634 Sundale Road, Massillon, OH'
   T: 'Drexdrex' | 'Ohio, Sundale Road, NULL, Perry Twp'
   p1=0.220 tset=0.22 ratio=0.27 atset=0.61 nm=0 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.06 rank=3 dup=0 tn=2.71 nwnew=1
[93970/8401105 India src3 p=0.537 pown=0.41 ntrue=3 npred=2 tp=2]
   S: 'Shelter Matrix Private Limited' | 'Madhopur, Bairiya Ward No-07, Madhopur, West Champaran, Bihar'
   T: 'Shelter Matrix' | 'No. 90 Madhopr, Madhopur, West Champaran, बिहार'
   p1=0.177 tset=1.00 ratio=1.00 atset=0.77 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=4 dup=0 tn=1.39 nwnew=0
[533/8846285 India src3 p=0.726 pown=0.614 ntrue=3 npred=2 tp=2]
   S: 'ZX Chit Private Limited' | '50/1112, D11 2Nd Floor, Muhammad Haji Building Station Junction, Edappally P.O, Kochi, Ernakulam, Kerala'
   T: 'ZX Private Limited Services' | 'D11 2Nd Floor, Muhammad Haji Building Station Junction, Edappally P.o, Kochi, 0/1112, കേരളം, Kerlala'
   p1=0.420 tset=0.44 ratio=0.44 atset=0.87 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=0.69 nwnew=1
[11710/9019382 US src3 p=0.643 pown=0.49 ntrue=4 npred=3 tp=3]
   S: 'Twisted Bakery' | '457 Kings Court, Akron, OH'
   T: 'Onyxarc' | '457 Kings Ct, Copley Twp, Ohio'
   p1=0.655 tset=0.19 ratio=0.19 atset=0.64 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.07 rank=4 dup=0 tn=1.39 nwnew=1
[147947/9053144 India src3 p=0.57 pown=0.181 ntrue=5 npred=3 tp=3]
   S: 'Delhi Impex Center' | 'A-6 Plot No- 2 Dharma Apartment Patparganj, I.P Extension, Delhi, East Delhi, Delhi'
   T: 'centerimpex.com' | 'Delhi, DL, A-6 Plot No- 2 Dharma Apartment Patcarganj, I.p Extension, East Delhi'
   p1=0.273 tset=0.00 ratio=0.00 atset=0.94 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.22 rank=6 dup=0 tn=1.79 nwnew=0
[19518/9143739 US src3 p=0.466 pown=0.344 ntrue=3 npred=0 tp=0]
   S: 'Bright Lloyds, Inc' | '1046 E Lake Road, Montville, CT'
   T: 'Bright Lloyds,' | '841 E Lake Rd, Oakdael, Connecticut'
   p1=0.052 tset=1.00 ratio=1.00 atset=0.60 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=1 dup=0 tn=1.39 nwnew=0
[59457/9265659 US src3 p=0.285 pown=0.193 ntrue=5 npred=4 tp=4]
   S: 'Guerrero Hall' | '1407 Garden Glen Lane, Pearland, TX'
   T: 'Guerrero Hall Co' | 'PMB 5730, Pearland, Texas, 1407 Garden Glen Lane'
   p1=0.950 tset=1.00 ratio=1.00 atset=0.94 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=2.08 nwnew=0
[91849/9757179 India src3 p=0.041 pown=0.018 ntrue=3 npred=2 tp=2]
   S: 'One Technology Limited' | '45E, 172/6, Hari Bhau Upadhyaya Nagar, Near Glitz Cinema, Ajmer, Rajasthan'
   T: 'Sri One Technology Límited' | ''
   p1=0.022 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.03 rank=5 dup=0 tn=4.04 nwnew=0
[25269/10103655 India src3 p=0.529 pown=0.376 ntrue=6 npred=5 tp=5]
   S: 'Uppal Mandal Consultancy Limited' | 'Telangana, Rangareddy District, Uppal Mandal, P-11/5/2, Sy No:50, Road No:8 Ida Nacharam'
   T: 'Uppal Mandal' | ''
   p1=0.362 tset=1.00 ratio=0.67 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.06 rank=7 dup=0 tn=4.80 nwnew=0
[118255/10319470 India src3 p=0.664 pown=0.541 ntrue=2 npred=1 tp=1]
   S: 'Blue Finance' | 'House No.105, Ward No.3, Near Dav School, Punjab, Firozpur, Jalalabad'
   T: 'Blue Finance' | 'House No.05, Mandi Amin Ganj,,punjab, Jalalabad, PB'
   p1=0.260 tset=1.00 ratio=1.00 atset=0.63 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.01 rank=2 dup=0 tn=1.79 nwnew=0
```
## Examples: FP (40)
```
[41248/24145 India src2 p=0.995 pown=0.991 ntrue=0 npred=1 tp=0]
   S: 'Mohite Works (India) Pvt. Ltd.' | 'No.15/13A, Prakash Avenue, Sembiam, Perambur, Chennai, Tamil Nadu'
   T: '... MOHITE LTD. (INDIA) GROUP PVT. WORKS' | 'DOOR NO 15/15A, PERAMBUR, CHENNAI, Tamil Nadu'
   p1=0.998 tset=1.00 ratio=0.67 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=1 dup=0 tn=2.64 nwnew=0
[88202/810820 US src2 p=0.994 pown=0.99 ntrue=5 npred=6 tp=5]
   S: 'Mureil Arriaga Valley Farmers' | '4430 Camden Circle, Dublin, OH, Bldg 33'
   T: 'Arriaga, Mureil Valley Farmers Co' | '4431 CAMDEN CIRCLE, DUBLLIN, OH'
   p1=0.467 tset=1.00 ratio=0.76 atset=0.98 nm=0 nc=1 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=0.69 nwnew=0
[24054/850079 India src2 p=0.84 pown=0.758 ntrue=3 npred=4 tp=3]
   S: 'Vkm Hospital' | '4/38, G.S.T. Road, Meenambakkam, Chennai-27, Tamilnadu, Tamil Nadu'
   T: 'Vkm Hospital Pvt Ltd' | 'B3/4/49, G.S.T. ROAD, MEENAMBAKKAM, CHENNAI-27, Tamil Nadu'
   p1=0.712 tset=1.00 ratio=1.00 atset=1.00 nm=1 nc=0 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=2.71 nwnew=0
[10285/1405601 US src2 p=0.992 pown=0.987 ntrue=2 npred=3 tp=2]
   S: 'N/U Colombier' | '329 Rustic Oaks Drive, Wentzville, MO'
   T: 'N/U Colombier' | 'W329S8485 OAK TREE DRIVE, WI, TOWN OF MUKWONAGO'
   p1=0.771 tset=1.00 ratio=1.00 atset=0.54 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.79 nwnew=0
[116604/1422215 India src2 p=0.99 pown=0.982 ntrue=3 npred=4 tp=3]
   S: 'Hari Tech Private Limited' | 'West Bengal, Calcutta, Kolkata, 3C Madan Street'
   T: 'হরি টেক প্রাইভেট লিমিটেড' | 'CALCUTTA, 3A, West Bengal, HOWRAH'
   p1=0.970 tset=0.06 ratio=0.06 atset=0.85 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=1 dom=0 legc=0 cmax=0.18 rank=3 dup=0 tn=2.20 nwnew=4
[102227/1970111 US src2 p=0.999 pown=0.998 ntrue=1 npred=2 tp=1]
   S: 'Marya N. Maciel, DMD, DDS PC' | '9078 Lucasburg Road, Byesville, OH'
   T: 'Lyrarizaevo' | '9078 LUCASBURG ROAD, BYESVILLE, OH'
   p1=0.986 tset=0.24 ratio=0.24 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.02 rank=2 dup=0 tn=3.78 nwnew=1
[19161/2203630 US src2 p=0.851 pown=0.775 ntrue=3 npred=4 tp=3]
   S: 'Winterset Urgent Care Center' | 'Winterset, 1011 3rd Avenue, IA'
   T: 'Winterset Urgent Care Center Group' | '1012 3RD AVE, DOUGLAS, IA'
   p1=0.576 tset=1.00 ratio=1.00 atset=0.56 nm=1 nc=0 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=4 dup=0 tn=2.40 nwnew=0
[1296/2305234 US src2 p=0.988 pown=0.981 ntrue=0 npred=1 tp=0]
   S: 'Crystal Energy Networks LLC' | 'Fayetteville, 3210 Friendly Road, NC'
   T: 'Group Crystal Energy Nétworks' | 'NC, FAYETTEVILLE, 3210 FRIENDLY RD'
   p1=0.995 tset=1.00 ratio=1.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=1 cmax=0.00 rank=1 dup=0 tn=1.61 nwnew=0
[116796/2730701 India src2 p=0.97 pown=0.951 ntrue=1 npred=2 tp=1]
   S: 'Vijay Projects LLP' | 'A-2, Green Park Extension, New Delhi, South Delhi, Delhi'
   T: 'विजय प्रोजेक्ट्स प्रा. लि.' | 'A-11, GREEN PARK EXTENSION, NEW DELHI, SOUTH DELHI, दिल्ली'
   p1=0.493 tset=0.05 ratio=0.05 atset=1.00 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=1 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.39 nwnew=4
[144438/2804398 US src2 p=0.865 pown=0.794 ntrue=4 npred=5 tp=4]
   S: 'Riane Rubright Safe Telephone Group' | '1840 Steel Road, Bee Branch, AR'
   T: 'SERNA RUBRIGHT TELEPHONE SAFE GROUP' | '1840-B STEEL RD, PO BOX 4856, BEE BRANCH, AR'
   p1=0.994 tset=0.90 ratio=0.72 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=1.95 nwnew=1
[54653/3075948 US src2 p=0.88 pown=0.815 ntrue=4 npred=5 tp=4]
   S: 'Piedmont Hcm PC' | '164 Crabapple Lane, Louisville, KY'
   T: 'Piedmont Hcm Co' | 'LOUISVILLE, 16 CRABAPPLE LANE, KY'
   p1=0.019 tset=1.00 ratio=1.00 atset=1.00 nm=0 nc=1 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=1 cmax=0.00 rank=5 dup=0 tn=2.48 nwnew=0
[111257/3272953 India src2 p=0.967 pown=0.946 ntrue=8 npred=9 tp=8]
   S: 'Kuchipudi Biotech Limited' | '7-50 Kuchipudi, Kuchipudi, Krishna, Andhra Pradesh'
   T: 'KUCHIPUDI BIOTECH BIOTECH GROUP LIMITED' | '7-7 KUCHIPUDI, KCHIPUDI, ఆంధ్రప్రదేశ్'
   p1=0.969 tset=1.00 ratio=1.00 atset=0.45 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=8 dup=0 tn=0.69 nwnew=0
[124315/3510567 US src2 p=0.94 pown=0.904 ntrue=3 npred=4 tp=3]
   S: 'Martin Connecticut LLC' | '3503 W 43rd St, Davenport, IA'
   T: 'ALLENCHENGMARTINEZ.COM' | '3503 WEST 43RD ST, IA, DAVENPORT'
   p1=0.659 tset=0.00 ratio=0.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.00 rank=4 dup=0 tn=2.94 nwnew=0
[3411/3695546 India src2 p=0.873 pown=0.805 ntrue=2 npred=3 tp=2]
   S: 'Seema Ventures Ltd' | 'Plot No 10, N-4, Cidco F Sector, Chikalthana, Aurangabad, Maharashtra, C/O Maitreya Ashok Mudkavi'
   T: 'Seema Services Limited' | 'NO #73 C/O MAITREYA ASHOK MUDKAVI, PLOT NO 10, N-4, CIDCO F SECTOR, CHIKALTHANA, AURANGABAD, Maharashtra'
   p1=0.889 tset=0.71 ratio=0.71 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=2.30 nwnew=1
[88341/4008669 US src2 p=0.726 pown=0.613 ntrue=0 npred=1 tp=0]
   S: 'Olvera College LLC' | '2342 Cave Mill Station Boulevard, Bowling Green, Unit 912, KY'
   T: 'pédiatricdentistryhighland.com' | '2342 CAVE MILL STATION BLVD, BOWLING GREEN, KY'
   p1=0.784 tset=0.00 ratio=0.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.00 rank=1 dup=0 tn=0.69 nwnew=0
[52402/4089862 India src2 p=0.841 pown=0.761 ntrue=2 npred=3 tp=2]
   S: 'Shiva Builders Private Limited' | '10, Hauz Khas Village, New Delhi, South Delhi, Delhi'
   T: 'SHIVA BUILDERS PRIVATE  LIMITED' | 'R-42, HAUZ KHAS, NEW DELHI, Delhi'
   p1=0.139 tset=1.00 ratio=1.00 atset=0.90 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.79 nwnew=0
[23079/4096820 India src2 p=0.899 pown=0.792 ntrue=6 npred=6 tp=5]
   S: 'Hisar Holdings' | 'Anne Maria Buildings, Thariode P.O, Vythiri, Wayanad, Kerala'
   T: 'Hisar Holdings' | ''
   p1=0.343 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.20 rank=6 dup=0 tn=3.71 nwnew=0
[77126/4747024 India src2 p=0.969 pown=0.949 ntrue=5 npred=6 tp=5]
   S: 'Bharati Agencies Care' | '8/1, L.N. Motilal Road Behala, Kolkata, Kolkata, Howrah, West Bengal'
   T: 'Bharati Agencies Care Private Limited' | '08/8, L.N. MOTILAL ROAD BEHALA, KOLKATA, KOLKATA, West Bengal'
   p1=0.995 tset=1.00 ratio=1.00 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=1 dup=1 tn=1.39 nwnew=0
[113746/5772758 US src3 p=0.999 pown=0.999 ntrue=6 npred=5 tp=4]
   S: 'Gibson, Bobette, MD' | '2966 Shasta Drive, Fl 1, Medford, OR'
   T: 'Gibson,-Bobette, SD' | '2966 Shasta Drive, NULL, Medford CITY, Oregon'
   p1=0.975 tset=0.94 ratio=0.94 atset=0.92 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.39 nwnew=1
[19426/5884608 US src3 p=0.954 pown=0.904 ntrue=4 npred=5 tp=4]
   S: 'Cam Vo Mckinley' | '525 Wister Way, Winthrop, WA'
   T: 'jcglass.com' | '##525 Wister Way, Winthrop, Washington'
   p1=0.593 tset=0.00 ratio=0.00 atset=0.92 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=1 legc=0 cmax=0.06 rank=5 dup=0 tn=2.94 nwnew=0
[2881/6109682 US src3 p=0.959 pown=0.933 ntrue=5 npred=6 tp=5]
   S: 'Pacific Wireless Consultants' | '272 Citation Drive, Virginia Beach City, VA'
   T: 'Wexbelodelta' | 'Virginia, Virginia Beach City, 272 Alameda Dr'
   p1=0.797 tset=0.20 ratio=0.35 atset=0.81 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=6 dup=0 tn=0.69 nwnew=1
[3441/6165970 US src3 p=0.996 pown=0.994 ntrue=4 npred=5 tp=4]
   S: 'Candra Bench, DPM, P.C.' | 'Long Island, 90 Leavitt Street, ME'
   T: 'Candra Bench, DPX, P.C.' | 'Maine, LONG Island, #90 Leavitt St'
   p1=0.977 tset=0.94 ratio=0.94 atset=0.94 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=2.48 nwnew=1
[128525/6726388 US src3 p=0.96 pown=0.936 ntrue=7 npred=8 tp=7]
   S: 'Pediatric Choice Physicians Group' | '4452 Whitmore Lane, Fairfield, OH'
   T: 'Choice Pediatric Group Physicians Group' | '453 Whitmore Ln, Farifield, Ohio'
   p1=0.490 tset=1.00 ratio=0.74 atset=0.92 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=8 dup=0 tn=2.83 nwnew=0
[96718/6917958 US src3 p=0.987 pown=0.978 ntrue=4 npred=5 tp=4]
   S: 'Tanitansy White, DPM, P.C.' | '7416 Ern Way, Wilmington, NC'
   T: 'Tanitansy Whit, DM, P.C.' | '##7416 Ern Way, Wilmington, North Carolina'
   p1=0.990 tset=0.94 ratio=0.94 atset=0.92 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=4 dup=0 tn=1.61 nwnew=2
[20163/6926731 India src3 p=0.643 pown=0.517 ntrue=0 npred=2 tp=0]
   S: 'DIO Plast Limited' | 'Plot No. 6 & 7, Flat No. 202, 2Nd Floor, Opp Prime Hospital, Manjeera Square, Ame, Erpet, Hyderabad, Telangana'
   T: 'Dpio Plast Limited' | 'No 75 & 7, Hyderabad, TG'
   p1=0.484 tset=0.95 ratio=0.95 atset=0.86 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.15 rank=2 dup=0 tn=4.01 nwnew=1
[134760/7415097 US src3 p=0.95 pown=0.92 ntrue=3 npred=4 tp=3]
   S: 'West Creative Safety' | '303 Lawson Avenue, Saint Paul, MN'
   T: 'West Creative' | '303 1/2 Cedar Street, Little Rock, Arkansas'
   p1=0.495 tset=1.00 ratio=0.79 atset=0.37 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=4 dup=0 tn=3.14 nwnew=0
[51255/7550622 India src3 p=0.99 pown=0.982 ntrue=4 npred=5 tp=4]
   S: 'Radha Cold Limited' | '504, Chiranjiv Tower, 43, Nehru Place, New Delhi, South Delhi, Delhi'
   T: 'Vantagetavoecto Sys' | 'Flat No. 504, Chiranjiv Tower, 43, Nehru Place, New Delhi, South Delhi, Delhi'
   p1=0.891 tset=0.21 ratio=0.28 atset=1.00 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.01 rank=4 dup=0 tn=3.04 nwnew=2
[129473/7776534 US src3 p=0.879 pown=0.814 ntrue=4 npred=5 tp=4]
   S: 'Gabbie Peth Innovative Voya L.L.C.' | '517 Holiday Avenue, Waxahachie, TX'
   T: 'Gabbie-Peth Innovative Voya LLC' | '52 Holiday Ave, Waxahachie, Texas'
   p1=0.428 tset=1.00 ratio=1.00 atset=0.94 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=1.61 nwnew=0
[85618/7974948 India src3 p=0.956 pown=0.929 ntrue=3 npred=4 tp=3]
   S: 'Artificial Multiworks' | '13, Jawahar Colony Radha Road Behind Sbi Adb Branch, Rampur, Uttar Pradesh'
   T: 'Artificial Multiworks Private Limited' | 'उत्तर प्रदेश, Rampur, Jawahar Colony Radha Road Behind Sbi Adb Branch, Rampur, G-15'
   p1=0.535 tset=1.00 ratio=1.00 atset=0.87 nm=0 nc=1 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.10 nwnew=0
[1128/8006733 India src3 p=0.96 pown=0.935 ntrue=2 npred=3 tp=2]
   S: 'Ss Estate Private Limited' | '1/1/1B Paikpara Row, Kolkata, Howrah, West Bengal'
   T: 'Ss Limited Éstate' | 'Door No 001/1/14B Paikpara Row, Howrah, Kolkata, WB'
   p1=0.968 tset=1.00 ratio=1.00 atset=0.95 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=2 dup=0 tn=3.09 nwnew=0
[62561/8135736 India src3 p=0.94 pown=0.904 ntrue=3 npred=4 tp=3]
   S: 'Sun Hitech Management Private Limited' | '1/1 Daryavihar Union Park Off Carter Road Khar West, Mumbai, Maharashtra'
   T: 'Sun Hitech Management Limited' | 'No 1/12 Daryavihar Union Park Off Carter Road Khar West, Mumbai, Mumbai (SUBURBAN), MH'
   p1=0.987 tset=1.00 ratio=1.00 atset=0.91 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.10 nwnew=0
[112117/8915583 India src3 p=0.977 pown=0.963 ntrue=4 npred=5 tp=4]
   S: 'SD Management Private Limited' | 'Sno 270, Kno 256, Beml Layout, Bangalore North, Bangalore, Karnataka'
   T: 'Sd Management Limited' | 'Sno 3-272, Bangalore North, Bangalore, KA'
   p1=0.507 tset=1.00 ratio=1.00 atset=0.93 nm=0 nc=1 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.01 rank=5 dup=0 tn=1.61 nwnew=0
[103305/8942842 US src3 p=0.996 pown=0.993 ntrue=4 npred=5 tp=4]
   S: 'Rocky Pinnacle Twelve LLC' | '1655 Frazier Street, Conroe, TX'
   T: 'Rocky Pinnacle Eight LLC' | 'Conroe, Texas, 1655 Frazier Street'
   p1=0.994 tset=0.82 ratio=0.78 atset=0.93 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=1.10 nwnew=1
[122776/9037951 US src3 p=0.907 pown=0.854 ntrue=4 npred=5 tp=4]
   S: 'White, Dombrowski and Curtiss Offshore Inc.' | '1436 Gibson Avenue, Indianapolis, IN'
   T: 'Inc. Torres, Dombrowski and Curtiss Offshore' | '143 Gibson Ave, Indianapolis, Indiana'
   p1=0.023 tset=0.90 ratio=0.90 atset=0.94 nm=0 nc=1 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=1.95 nwnew=1
[102657/9038479 India src3 p=0.998 pown=0.997 ntrue=3 npred=4 tp=3]
   S: 'Tvs Institute-<CITY_NAME>' | 'House No. 05 T1, Nilanga Tq- Latur, Latur, Maharashtra'
   T: 'Tvs  Institute-<CITY_NAME> Limited' | 'Huose No. D/05 T5, Latur, MH'
   p1=0.991 tset=1.00 ratio=1.00 atset=0.53 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=1 dup=0 tn=3.40 nwnew=0
[129046/9205311 India src3 p=0.689 pown=0.57 ntrue=0 npred=1 tp=0]
   S: 'Modern Technology Private Limited' | 'Pune, Flat No. 14, G-Bld, Dhanraj Park, Wakad, S. No. 207/3, Maharashtra, Pune'
   T: 'मॉडर्न टेक्नोलॉजी लिमिटेड' | 'S. No. 207/8/3, Flat No. 14, G-bld, Dhanraj Park, Wakad, Pune, महाराष्ट्र'
   p1=0.964 tset=0.05 ratio=0.05 atset=0.83 nm=1 nc=0 near=0 tbl=0 sbl=0 ind=1 dom=0 legc=0 cmax=0.00 rank=1 dup=0 tn=2.56 nwnew=3
[39700/9370273 US src3 p=0.997 pown=0.996 ntrue=1 npred=2 tp=1]
   S: 'Atlantic Harbor Center' | '106 Sheridan Street, IL, Watseka'
   T: 'Atlantic Center Harbor' | ''
   p1=0.987 tset=1.00 ratio=0.68 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=2 dup=0 tn=0.69 nwnew=0
[79689/9619681 US src3 p=0.999 pown=0.999 ntrue=2 npred=3 tp=2]
   S: 'Dock, Hileman & Calhoun LLC' | '1306 Rosemary Avenue, Durham, NC'
   T: 'Dock Hileman + Calhoun LLC' | ''
   p1=0.995 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=4.96 nwnew=0
[89739/9831854 US src3 p=0.95 pown=0.92 ntrue=2 npred=3 tp=2]
   S: 'Internal Medicine Clean Partners' | '17442 663, Farmersville, TX'
   T: 'Internal Medicine Clean Partners Partners' | '17443 663, Farmersville, Texas'
   p1=0.589 tset=1.00 ratio=1.00 atset=0.91 nm=1 nc=0 near=1 tbl=0 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=3 dup=0 tn=1.79 nwnew=0
[47915/10223979 US src3 p=0.986 pown=0.977 ntrue=4 npred=5 tp=4]
   S: 'Lynx Trading Inc' | '321 Runaway Bay Circle, Unit 1D, Mishawaka, IN'
   T: 'Lynx Inc. Trading' | ''
   p1=0.978 tset=1.00 ratio=1.00 atset=0.00 nm=0 nc=0 near=0 tbl=1 sbl=0 ind=0 dom=0 legc=0 cmax=0.00 rank=5 dup=0 tn=1.79 nwnew=0
```
## Examples: blocking misses (40)
```
[95881/7515 India src2 p=None pown=None ntrue=5 npred=1 tp=1]
   S: 'One Logistics Private Limited' | 'Bengaluru, Bangalore, No. 1/1, Karnataka, 4Th Cross.(Old No, 1, 32Nd Cross), 7Th Block (W), Jayanagar'
   T: 'ಒನ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್' | 'NO. B3/1/1, 4TH CROSS.(OLD NO, 1, 32ND CROSS), 7TH BLOCK (W), JAYANAGAR, BANGALORE, ಕರ್ನಾಟಕ'
[113746/13192 US src2 p=None pown=None ntrue=6 npred=5 tp=4]
   S: 'Gibson, Bobette, MD' | '2966 Shasta Drive, Fl 1, Medford, OR'
   T: '*** GIBSON, BOMBCT,E MD' | '296 SHASTA DR, MEDFORD, OR'
[96660/533366 India src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Sree Projects Private Limited' | '2Nd Floor, No.34, 1St Main, 1St Cross Koramangala 1St Block, Bangalore, Karnataka'
   T: 'ಶ್ರೀ ಪ್ರಾಜೆಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್' | 'NO 2ND FLOOR, BANGALURU, BANGALORE, Karnataka'
[104662/699029 India src2 p=None pown=None ntrue=3 npred=1 tp=1]
   S: 'Great Consultancy Private Limited' | 'No.70, 5Th Cross, 5Th Main, N.R. Colony Bangalore - 19. 560 019., Bangalore, Karnataka'
   T: 'ಗ್ರೇಟ್ ಕನ್ಸಲ್ಟೆನ್ಸಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್' | 'N.R. COLONY BANGALORE - 19. 560 019., BANGALORE, Karnataka, #70'
[116633/1081162 India src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Shakti Investments Limited' | 'Plot No.83, Electronics City, Bangalore South ., Karnataka'
   T: 'ಶಕ್ತಿ ಇನ್ವೆಸ್ಟ್\u200cಮೆಂಟ್ಸ್ ಲಿಮಿಟೆಡ್' | 'BANGALORE SOUTH ., PLOT NO.083, Karnataka, BANGALORE SOUTH .'
[25853/1182432 US src2 p=None pown=None ntrue=6 npred=5 tp=5]
   S: 'Valley Project' | '4332 Meredith Avenue, Omaha, NE'
   T: 'Valley Prómte' | '433 MEREDITH AVENUE, OMAHAA CITY, NE'
[102794/1377719 India src2 p=None pown=None ntrue=4 npred=3 tp=3]
   S: 'Universal Ventures Private Limited' | 'Tamil Nadu, Ga & Gb, No.11, Chennai, Fourth Main Road Ext Kotturpuram, Riviera Park'
   T: 'யுனிவர்சல் வென்ச்சர்ஸ் பிரைவேட் லிமிடெட்' | 'BLOCK C-779. RIVIERA PARK, FOURTH MAIN ROAD EXT KOTTURPURAM, CHENNAI, Tamil Nadu'
[16611/1813630 India src2 p=None pown=None ntrue=2 npred=1 tp=1]
   S: 'Design Switchgear Pvt Ltd' | '305 Shrusti Residency, Sector 18, Navi Mumbai, Khalapur, Raigarh(Mh), Maharashtra'
   T: 'DESIGN SGWITCGEAR PVT LTD' | ''
[79871/2093909 India src2 p=None pown=None ntrue=3 npred=1 tp=1]
   S: 'Gold Logistics Private Limited' | 'Krishna Industrial Complex, No-5, Uttarahalli Hobli B-61, Subramanyapura, Bangalore, Karnataka'
   T: 'ಗೋಲ್ಡ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್' | 'Karnataka, KRISHNA INDUSTRIAL COMPLEX, BANGALORE'
[60089/2174533 US src2 p=None pown=None ntrue=5 npred=4 tp=4]
   S: 'Express Agricultural Holdings' | '12244 Jonathan View Lane, Draper City (sl Co), UT'
   T: 'Express Agiriclutuila Holdings' | 'Jonathan View Ln, DRAER, UT'
[11514/2485169 US src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'National Products Enterprises' | '709 Lincoln Street, Topeka, KS'
   T: 'National Products LLC' | 'LINCOLN ST, TOPEKA, KS'
[94909/2674860 US src2 p=None pown=None ntrue=4 npred=3 tp=3]
   S: 'Physical Therapy Atlantic Physicians PLLC' | '427 Red Star Road, Oak Hill, WV'
   T: 'Physical Therapy Atlantic Phyicdcns PLLC' | 'RED STAR ROAD, <NULL>, OAK HILL, WV'
[132601/2758699 India src2 p=None pown=None ntrue=6 npred=2 tp=2]
   S: 'Guru Producer Pvt Ltd' | 'L-2/81, Second Floor, New Mahavir Nagar, New Delhi, West Delhi, Delhi'
   T: 'गुरु प्रोड्यूसर प्रा. लि.' | 'L-2/81, NEW DELHI, WEST DELHI, दिल्ली'
[104656/3018880 US src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Dermatology Bright Health LLC' | '24 NW 4 St, Le Mars, IA'
   T: 'LLC Dermatology Bright Hebrth' | '24. NORTHWEST 4 ST, LEM ARS, IA'
[30893/3147932 US src2 p=None pown=None ntrue=6 npred=4 tp=4]
   S: 'Elliott Biopharmaceuticals, LLC' | '321 King Street, NC, Ayden'
   T: 'Elliott  LLC Services' | ''
[48939/3505211 US src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Lower Montessori School' | 'Whitestown, NY, 18 Chateau Drive'
   T: 'lowermontessorischool.com' | '1 CHATEAU DRIVE, WHITESTOWN, NY'
[28921/3577793 India src2 p=None pown=None ntrue=4 npred=1 tp=1]
   S: 'Gujarat Hospitality Pvt Ltd' | 'C/O Lt. Ramdas Shamshabad, Azamgarh, Uttar Pradesh'
   T: 'गुजरात हॉस्पिटैलिटी प्रा. लि.' | 'C/O LT. RAMDAS SHAMSHABAD, AZAMGARH, Uttar Pradesh'
[146591/3659930 US src2 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Edwards Frontier Coca Inc' | '980 Cleveland Avenue, Unit 418, Columbus, OH'
   T: 'Edwards Fdeotfr Coca-Inc' | '98 CLEVELAND AVE, COLUMBUS, OH'
[83922/4047637 US src2 p=None pown=None ntrue=5 npred=4 tp=4]
   S: 'UO Ace Solutions' | '265 Wildwood Road, Lenoir, NC'
   T: 'U0 Ace Solutions Inc.' | '65 WILDWOOD RD, LENORI, NC'
[126661/4902091 US src2 p=None pown=None ntrue=4 npred=2 tp=2]
   S: 'Campbell and Martin Inc.' | '11002 Maple Rock Drive, Humble, TX'
   T: 'Campbell and-Martin Inc. Service' | '1100 MAPLE ROCK DR, HUMBLE, TX'
[110541/5065698 US src3 p=None pown=None ntrue=7 npred=5 tp=5]
   S: 'Continental Family Practice' | '102 Elkmont Drive, VA, Elkton Town'
   T: 'Continental  Family' | 'Elkmont Dr, Elkton Town, Virginia'
[41511/5104571 US src3 p=None pown=None ntrue=3 npred=2 tp=2]
   S: "Martin's Signature Wholesale, Inc" | '121 High Point Road, Deer Lodge, TN'
   T: "Martin's Signature" | 'High Point Rd, DEER Lodgec ITY, Tennessee'
[115765/6193333 US src3 p=None pown=None ntrue=2 npred=1 tp=1]
   S: 'Molina Alussa' | '4207 4th Avenue, San Bernardino, CA'
   T: 'Molina Sérvices' | ''
[99498/6215340 US src3 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Cinder Foods Inc' | 'Hanover Park, IL, 721 Haddam Way'
   T: 'cinderfoods.com' | '719 Haddam Way, Illinois, Hanover Park'
[41164/6250237 US src3 p=None pown=None ntrue=7 npred=6 tp=6]
   S: 'Collins Classic Oyj LLC' | '6722 S Jordan Parkway, South Jordan, UT'
   T: 'Collins Oyj  LLC Services' | 'Utah, SOUT Jordan, S Jordan Parkway'
[125229/6769058 US src3 p=None pown=None ntrue=6 npred=4 tp=4]
   S: 'Cozy Cleaning Service' | '9174 38th Avenue, Rolette, ND'
   T: 'Cozy Cleaning Service Corporation Enterprises' | ''
[65668/7364065 US src3 p=None pown=None ntrue=3 npred=2 tp=2]
   S: 'Solstice' | '12 Birch Street, Anaconda-deer Lodge, MT'
   T: 'Sólstice Enterprises' | ''
[143449/7374539 US src3 p=None pown=None ntrue=2 npred=1 tp=1]
   S: 'Dental Allied Center' | '348 13th, Paris, TX'
   T: 'Dental-Allied Centre' | ''
[72296/7509543 US src3 p=None pown=None ntrue=6 npred=5 tp=5]
   S: 'Smith Frontier Information' | '3650 Ridge Road, Unit LOT 98, Gary, IN'
   T: 'Smith Frontier Services' | ''
[129034/7872640 US src3 p=None pown=None ntrue=6 npred=5 tp=5]
   S: "Martina's Auto Repair Corp" | '2103 Arthur Street, Wichita Falls, TX'
   T: 'Martinasautorepair.Com' | 'null, 2101 Arthur Street, Wichita Falls, Texas'
[75837/7940082 US src3 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Quality Better Machines LLC' | 'IL, Rockford, 307 Oakley Avenue'
   T: 'Quality Better' | '07 Oakley Avenue, Rockford, Illinois'
[138483/8000275 India src3 p=None pown=None ntrue=4 npred=3 tp=3]
   S: 'Jay Estate Limited' | 'No.20/2, Sundar Nagar, Poonamallee, Chennai, Tamil Nadu'
   T: 'ஜெய் எஸ்டேட் லிமிடெட்' | 'No.20/2, Chennai, TN'
[135248/8426563 India src3 p=None pown=None ntrue=3 npred=1 tp=1]
   S: 'South Impex Private Limited' | 'No.3/1, Gangadhar Chetty Road, Bangalore, Karnataka'
   T: 'southimpex.com' | 'Bangalore, KA, No./1, Bangalore, Gangadhar Chetty Road'
[83286/8471149 India src3 p=None pown=None ntrue=3 npred=2 tp=2]
   S: 'Ganga Group of Companies' | '604- B, Crystal Plaza, 6Th Floor M R No-2, Link Road, Opposite Infenity Mall Andher, I West, Mumbai, Mumbai City, Maharashtra'
   T: 'Ganga Group of Partners' | 'Hn 354 604- B, Greater Bombay, Mumbai City, MH'
[25990/8568595 US src3 p=None pown=None ntrue=4 npred=2 tp=2]
   S: 'Cardiology Legacy Care Associates' | '10219 River Road, Potomac, MD'
   T: 'Cardiology Legacy Legacy Care' | 'River Road, null, Potomac, Maryland'
[37538/8717127 US src3 p=None pown=None ntrue=5 npred=2 tp=2]
   S: 'Vior' | '15 Park Circle, Brookfield, VT'
   T: 'Vior.Com' | '15 Park Cir, Brookfield, Vermont'
[88984/8741804 US src3 p=None pown=None ntrue=4 npred=1 tp=1]
   S: 'Pediatric Dental Specialists Inc.' | 'OR, 242 14th Avenue, Eugene, Unit Apartment 3'
   T: 'Pediatric Dental  Specialists' | '14th Avenue, Unit Apartment 3, Euggene, Oregon'
[125513/9139880 India src3 p=None pown=None ntrue=6 npred=5 tp=5]
   S: 'Bangalore Distribution LLP' | '# 677, 1St Floor, 27Th Main, 13Th Cross, Hsr Layout 1St Sector, Bangalore, Karnataka'
   T: 'Evozeph' | '#677, 1St Floor, 27Th Main, 13Th Cross, Hsr Layout 1St Sector, Bangalore, KA'
[92072/9946853 India src3 p=None pown=None ntrue=5 npred=4 tp=4]
   S: 'Seven Logistics Private Limited' | 'Maharashtra, Sarvadnya Apartment, F No 3P, No 4 Borkute Lay, Near, Nagpur'
   T: 'सेवन लॉजिस्टिक्स प्राइवेट लिमिटेड' | 'Sarvadnay Apartment, Nagpur, Nagpur Kingsway, महाराष्ट्र'
[46201/9950295 India src3 p=None pown=None ntrue=5 npred=3 tp=3]
   S: 'Dk & Sons Private Limited' | 'No.63, Katha No.6, Agara, Tataguni Post, Kengeri, Bangalore South, Bangalore, Karnataka'
   T: 'Pyraciravio' | 'No 063, Bangalore South, Bangalore, KA'
```
