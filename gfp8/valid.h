/*
Copyright 2020, Yves Gallot

gfp8 is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <inttypes.h>

namespace private_ulll
{
	template<__uint128_t V> constexpr __uint128_t eval() { return V; }
	template<__uint128_t V, char C, char... Cs> constexpr __uint128_t eval() { return eval<10 * V + C - '0', Cs...>(); }
	template<char... Cs> constexpr __uint128_t operator "" _ulll() { return eval<0, Cs...>(); }
}

template<char... Cs> constexpr __uint128_t operator "" _ulll() { return ::private_ulll::operator "" _ulll<Cs...>(); }

const size_t test_size = 574;
static const __uint128_t b_test[test_size] = { 382897414818370ull, 5662378212145050ull, 13532961108287370ull, 13680834962835706ull, 20175421633472220ull,
	21532779520850890ull, 30864181565894616ull, 32464193790674490ull, 35849989409425906ull, 46379602423369306ull, 48469738856081046ull, 53580089311869376ull,
	54252753717381336ull, 58294094742729160ull, 70319090254411386ull, 70509959981244616ull, 76191370377166980ull, 80522908101098346ull, 81869833812543480ull,
	87125356884407650ull, 93533743213366080ull, 94241316812951616ull, 96069633013625310ull, 98734492485262926ull, 99982546618832460ull, 100599182049888616ull,
	101888385807349570ull, 122490613014891690ull, 132444940489668226ull, 148132441845673110ull, 148181524279237230ull, 155503343431292766ull, 158760058246800840ull,
	160086573010628236ull, 166514921921417196ull, 182257405713197536ull, 187184291402291170ull, 195754021695907260ull, 201207409965440820ull, 203526025988672896ull,
	206426376830479920ull, 208573677797758290ull, 215189959017565350ull, 216481912155555546ull, 223702795105826520ull, 229411323686783070ull, 233032322396230296ull,
	233141413802066760ull, 242864873700451476ull, 244533113310497940ull, 245861667908543206ull, 254025114516231990ull, 268469888692280860ull, 270823113642446856ull,
	278548159646886606ull, 279802738650263700ull, 283134538192087866ull, 292682483138149336ull, 302267774482888926ull, 308117747508311076ull, 309099054205426996ull,
	316546827266851480ull, 321161378268680626ull, 321647107622768950ull, 322897254026783436ull, 331831753983560736ull, 333954241398116220ull, 339126820107067066ull,
	377276159236066630ull, 378464691075467146ull, 382573398156155376ull, 392775445051721590ull, 410848394485758916ull, 447388949405176776ull, 448557986291645146ull,
	464074207539984730ull, 488682321579382200ull, 490196815395979056ull, 506486211021787936ull, 512589246925984950ull, 523738742397668266ull, 524491666972404870ull,
	524575206228107490ull, 530975397826424040ull, 532830729795285766ull, 533419520524516390ull, 552571517961200160ull, 557260603556255380ull, 573535342161893016ull,
	576456072339791956ull, 578152287227938680ull, 595919930055619500ull, 598395387953215576ull, 605528810208646366ull, 610456986567961326ull, 611286329119196836ull,
	613975965476702736ull, 621461821526125980ull, 624881450149097880ull, 627468323045897460ull, 629350051064184340ull, 630539631659541186ull, 660426443148323316ull,
	670045561309680970ull, 674079355507640496ull, 675480347340218460ull, 680931400173618610ull, 698271336309513216ull, 703319520860359846ull, 703815205220729686ull,
	720775203036926236ull, 728193675505628580ull, 740946236909868766ull, 748628726481095986ull, 774955032656638230ull, 797830970393058570ull, 807955026356095776ull,
	817589013835412956ull, 817779334526013406ull, 821889172841511456ull, 834470663273207950ull, 858161353175848390ull, 859663903108361550ull, 864715123612815316ull,
	865245576730494096ull, 867361827309416056ull, 882298343365184826ull, 882956989106714596ull, 939727861388489860ull, 950480132634031116ull, 953936380820984130ull,
	956225576290889346ull, 987339034412394876ull, 988199703524941860ull, 1015828522993056606ull, 1026971789909986410ull, 1037493040226425270ull, 1051436146380103146ull,
	1053772595398554670ull, 1093941850199777386ull, 1112259995212747066ull, 1121993983692794550ull, 1154298700137919680ull, 1173972430894984680ull, 1177201242034762600ull,
	1181042534059648110ull, 1183487241733002300ull, 1217383255226380060ull, 1222937676037451146ull, 1225236567261391650ull, 1228896106242376810ull, 1234327620750735310ull,
	1259351774688096610ull, 1266888783911592936ull, 1276722378788309950ull, 1277253356382441070ull, 1278801264068469246ull, 1285414142646617880ull, 1301971220240918400ull,
	1312049287895828860ull, 1320674350499087560ull, 1325560767179438836ull, 1328853872788578766ull, 1342377795280230126ull, 1350734156094532590ull, 1411943162027648140ull,
	1415919120654319240ull, 1420805092011826530ull, 1426542264671472550ull, 1433504468459569320ull, 1444557513131441850ull, 1446878436924418866ull, 1453410744603168576ull,
	1459599120637535746ull, 1463988080740657816ull, 1473536034608005006ull, 1480197569776969026ull, 1481954686499155710ull, 1485679833002380026ull, 1492702127493341730ull,
	1500336403767916390ull, 1563922376937740550ull, 1564608861287994646ull, 1574941180250201670ull, 1584336223986460890ull, 1584890511509815480ull, 1642475428133532340ull,
	1681365185421230796ull, 1686851291525298220ull, 1687512913658039310ull, 1725610806261404190ull, 1744393162992690856ull, 1751324292863382690ull, 1807525142475134710ull,
	1817486419755677476ull, 1852135568771082900ull, 1867812648402219360ull, 1890639538780930050ull, 1919122597840678566ull, 1922916458545813020ull, 1927095966446063466ull,
	1931041704606216120ull, 1939454694720848286ull, 1951933944025510126ull, 1957583736059240100ull, 1957704328893328090ull, 1963856411808447436ull, 1970520995526316686ull,
	1982890184659780380ull, 1989147935211116500ull, 2001221506097547346ull, 2065517020297366750ull, 2068527383570488050ull, 2118207298469995566ull, 2120918181306307530ull,
	2137337195853136170ull, 2152568338817553960ull, 2171709289530460986ull, 2172551675705714430ull, 2177797145085216106ull, 2181563297147558346ull, 2182666452864671670ull,
	2190610898894368486ull, 2192490616691082340ull, 2197142718532481310ull, 2210757129958834986ull, 2227316751878121046ull, 2241963022271488686ull, 2257403624463997230ull,
	2259779257451682246ull, 2269157260131632280ull, 2277215099851172176ull, 2325392038745966400ull, 2331786860741282220ull, 2335667844213826276ull, 2365114417125943390ull,
	2382177488244016680ull, 2385001231093932270ull, 2409980384733853230ull, 2414009566148625000ull, 2424722639864240640ull, 2425340773332188806ull, 2437997929364176020ull,
	2446303007336142000ull, 2466001863449001810ull, 2475092159632961836ull, 2485149631394955460ull, 2492073711534360786ull, 2495200012609212730ull, 2495247224468136526ull,
	2503208942953908900ull, 2506363537607018446ull, 2519970297866607280ull, 2529242800914669690ull, 2537480222853303010ull, 2546615480278611010ull, 2547181547700451246ull,
	2552136766873990710ull, 2580307219688400310ull, 2581471129275096276ull, 2582314727506182870ull, 2584328495948574966ull, 2594493296729530980ull, 2621091933620549796ull,
	2621584515468682270ull, 2634581477486735110ull, 2649865980625543086ull, 2661673216076624526ull, 2679896942186559630ull, 2697460936261797570ull, 2704809627660711876ull,
	2714728742521092840ull, 2715179275455604870ull, 2727372929336354640ull, 2741783555622218256ull, 2766298185733742686ull, 2790663015696013056ull, 2797775207131657806ull,
	2798192826496864186ull, 2798393962422402150ull, 2818744144364485366ull, 2834980124005474120ull, 2841324482644529116ull, 2853562380208871706ull, 2902786929858333960ull,
	2905928092256847660ull, 2910626365147968406ull, 2929168508277709600ull, 2932100405424440940ull, 2943649087109661976ull, 2950637453767116076ull, 2959045708198020460ull,
	2959513444890919530ull, 2972943559399124196ull, 3007727114391947266ull, 3007935654873244800ull, 3015309600555850690ull, 3034628836067568250ull, 3035269005168814300ull,
	3044203008633650550ull, 3048621217188695346ull, 3051161458566767550ull, 3063202073936673996ull, 3066861535342933896ull, 3077999801854698906ull, 3086475747772759546ull,
	3091753665303258096ull, 3093807683550879090ull, 3094936309832281206ull, 3099958131016140310ull, 3118222738978400106ull, 3118773473911726426ull, 3139136404557165580ull,
	3155071483645686720ull, 3163628717065324810ull, 3169596919322955390ull, 3189142081799013940ull, 3191856261214775646ull, 3211802863489948990ull, 3216761139170684466ull,
	3218667527175608376ull, 3222977218133622580ull, 3235048887426056916ull, 3235352986315278946ull, 3249187458833431410ull, 3281048514127236766ull, 3284150557259779650ull,
	3284678120550113796ull, 3287832834769974016ull, 3295887489155052300ull, 3321080620620830860ull, 3323040513927158046ull, 3336451928026050480ull, 3348781637421447496ull,
	3350743621657364796ull, 3355795539911694240ull, 3358122424494564456ull, 3376314944870915170ull, 3383631381559921020ull, 3392905134967348696ull, 3399504231321473250ull,
	3403698708135094710ull, 3406973298898400310ull, 3417843562985063520ull, 3429661782237736620ull, 3441333043831927806ull, 3442685976097684816ull, 3450025545995163990ull,
	3450142564012899190ull, 3455011222018566250ull, 3468090416378348820ull, 3470309242647389880ull, 3504878125681756470ull, 3557420029414147270ull, 3581028288361438756ull,
	3602682203166506880ull, 3619736932990743150ull, 3623739432340634280ull, 3649135845495398616ull, 3672971190417227706ull, 3698380049657756230ull, 3698607549519555996ull,
	3704229687451818120ull, 3709806691633850526ull, 3711504443726484676ull, 3732354109549388866ull, 3792668560433573880ull, 3808690654578814930ull, 3816944333949220966ull,
	3824159152106617720ull, 3825059390272106256ull, 3830117003912087346ull, 3831152948429873836ull, 3838916433602704300ull, 3850390339850998606ull, 3861989922748656160ull,
	3863306934406029660ull, 3875493064507561560ull, 3900240465667795056ull, 3963674001421587876ull, 3972154065430306936ull, 3974868692963967526ull, 3988233484085440716ull,
	3988804835558583796ull, 3999603837542774580ull,

	5015866531854994296ull, 5020039094699418046ull, 5024908825116956730ull, 5027659764944912190ull, 5030673428003793390ull, 5033041441414481766ull, 5039710762907958310ull,
	5048768577450546510ull, 5053175281883158920ull, 5053798333882469640ull, 5069946996748688370ull, 5127789214363125040ull, 5148233856086227476ull, 5150915158434201556ull,
	5158843169271642946ull, 5159961616674816816ull, 5199308626612982376ull, 5205366766124708236ull, 5232742995329310106ull, 5232998059885834086ull, 5234426860635582766ull,
	5237279774664091030ull, 5247272122398210466ull, 5257317580140602460ull, 5259186025676315850ull, 5268658578067702216ull, 5278093979795024196ull, 5292882782815427406ull,
	5298476654846988076ull, 5307205689677960586ull, 5371395434253444976ull, 5383173129928290726ull, 5390163248988909576ull, 5409641903519758186ull, 5413014795345061920ull,
	5416219033555954110ull, 5428574284004836630ull, 5430295664806880046ull, 5435003722592433136ull, 5448833414204789070ull, 5454448441697002380ull, 5463853883276529156ull,
	5475402214573972986ull, 5481773878473805426ull, 5484797627048484550ull, 5486135747050059216ull, 5499961040106643380ull, 5500581703692108996ull, 5513670977350958436ull,
	5516810969977316740ull, 5518352468489803780ull, 5528211909933681960ull, 5540314120193697420ull, 5545721650237963636ull, 5558351814788226066ull, 5566992767606519926ull,
	5578849369550088966ull, 5595841928126374096ull, 5644644136162632430ull, 5701957840048365036ull, 5764481760467704146ull, 5766860907032914836ull, 5783249550842797950ull,
	5799016292529689556ull, 5812935804159209460ull, 5835065598693436600ull, 5838269468683447000ull, 5845597740757823686ull, 5870799045859390630ull, 5886414984934993276ull,
	5919033439263658710ull, 5951612498830722276ull, 5952985780350584220ull, 5957470835293543230ull, 5963967380139223266ull, 5968775781261749886ull, 5974435612410282700ull,
	5985186844300192860ull, 5991748217315976480ull, 5999648229718394290ull, 6034304821926902020ull, 6034852614112339866ull, 6046592983823156520ull, 6068495951358078150ull,
	6074291185128805290ull, 6088866310441375296ull, 6106050847592493246ull, 6111620945656450276ull, 6118776602567059380ull, 6145784338907686656ull, 6156070167360479650ull,
	6187056512203644276ull, 6191247564365529270ull, 6200537068274006560ull, 6232065387854815096ull, 6254392414878868570ull, 6276951772360005240ull, 6284199815932366266ull, 
	6289305437466166206ull, 6289455850717664686ull, 6299888821436250796ull,

	11724428404256614330ull, 11727559285985399650ull, 11730005955430718476ull, 11740684768605843870ull, 11760564978425382990ull, 11777828478979473766ull, 11792315519159656770ull,
	11797780815096214656ull, 11805075093443465680ull, 11819113529708569740ull, 11824925374798306576ull, 11826335525595088110ull, 11831236855748074956ull, 11841228568604043906ull,
	11855423881184764560ull, 11861482224569066206ull, 11865082711671444840ull, 11871204042791642896ull, 11881915603681022166ull, 11898506871291674760ull, 11899899667710467836ull,
	11907446925768906450ull, 11918582706616475106ull, 11926464821166034746ull, 11928017732228901240ull, 11937218114951262766ull, 11979112684767091000ull, 12010978667448934570ull,
	12016772643140028780ull, 12069882107324605246ull, 12128066697019334110ull, 12141868067922473406ull, 12144054219806109150ull, 12147272564447978500ull, 12162602728338888030ull,
	12165040724666823406ull, 12195547330681204596ull, 12202613623406368146ull, 12202913298891093640ull, 12227390235604356940ull, 12231595387761651510ull, 12294405786848312436ull,
	12296993369674713190ull, 12338350873047977310ull, 12345952393847318286ull, 12347604020964599046ull, 12400685211607101796ull, 12408748773526436656ull, 12422827462198594950ull,
	12442090311432734806ull, 12451028873859303526ull, 12459370688521064640ull, 12475699472019158886ull, 12477977396721304570ull, 12479853129492832906ull, 12494988074949548766ull,
	12509745189838680730ull, 12523419205631674890ull, 12540587839612783206ull, 12554013216678204316ull, 12578125708304085840ull, 12604434837346077646ull, 12627239070735316066ull,
	12634755202503816256ull, 12650082458794959666ull, 12664867230629892586ull, 12665286402453030090ull, 12674067850207917946ull, 12685066690670577976ull, 12698770725437134080ull,
	12701022957709436040ull, 12712362101201832916ull, 12724097292070193850ull, 12790526501740973650ull, 12800001430269747906ull, 12808266866256159690ull, 12816692695997529640ull,
	12823857558524024190ull, 12824593286978999970ull,

	25019747681899499340_ulll, 25034695668913233156_ulll, 25062838236257448480_ulll,

	39614081257130190374474593146_ulll,

	240164550712338756ull, 3686834112771042790ull, 6470860179642426900ull, 7529068955648085700ull, 19344979062504927000_ulll
};
