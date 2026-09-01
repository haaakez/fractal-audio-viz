"""Curated, source-attributed Mandelbrot deep-zoom locations.

Coordinates stay as decimal strings all the way into MPFR.  The conjugate
entries are exact reflections across the real axis; the Mandelbrot set has
that symmetry, so they retain the source location's usable zoom depth.
"""

from __future__ import annotations

from dataclasses import dataclass


MDZ_GALLERY_URL = "https://mathr.co.uk/mdz/gallery/"
FRACTALSHARK_SOURCE_URL = (
    "https://github.com/mattsaccount364/FractalShark/"
    "blob/main/HpSharkFloatLib/FractalViewPresets.cpp"
)


@dataclass(frozen=True)
class DeepZoomPoint:
    slug: str
    name: str
    x: str
    y: str
    source_log10_zoom: float
    source_url: str
    source_name: str
    screened_log10_zoom: float = 150.0
    conjugate_of: str | None = None
    formula: str = "mandelbrot"
    # A Julia preset is defined by both its viewport centre and its fixed c.
    # Other formula families leave this unset.
    julia_c: tuple[str, str] | None = None


def _negated(value: str) -> str:
    return value[1:] if value.startswith("-") else f"-{value}"


def _pair(
    slug: str,
    name: str,
    x: str,
    y: str,
    source_log10_zoom: float,
    source_url: str,
    source_name: str,
    *,
    screened_log10_zoom: float = 150.0,
) -> tuple[DeepZoomPoint, DeepZoomPoint]:
    original = DeepZoomPoint(
        slug,
        name,
        x,
        y,
        source_log10_zoom,
        source_url,
        source_name,
        screened_log10_zoom,
    )
    conjugate = DeepZoomPoint(
        f"{slug}-mirror",
        f"{name} (mirror)",
        x,
        _negated(y),
        source_log10_zoom,
        source_url,
        source_name,
        screened_log10_zoom,
        slug,
    )
    return original, conjugate


# The four named locations are the exact centres from the corresponding KFR
# files in Claude Heiland-Allen's MDZ gallery.  The remaining ten are centres
# of built-in deep test views in FractalShark.  They were screened with this
# renderer at e150; ``screened_log10_zoom`` keeps random selection away from a
# view that is visually all-interior before its useful depth.
DEEP_ZOOM_POINTS = tuple(
    point
    for pair in (
        _pair(
            "oldwooddish",
            "Old Wood Dish",
            "-7.621179376797638683152412470302754234697554266277384201803182452707988048871166668157739268919140091423417335237629214081127371665327888360528335010378835950994472786588199757e-1",
            "-9.573175593807915218110127195876981729185284295753173780078525497988166932168660068170631431015884613850090347745780605812248444859260422571076360932193737688403082995582081506e-2",
            152.3476,
            MDZ_GALLERY_URL,
            "oldwooddish.kfr",
        ),
        _pair(
            "xx10-star",
            "XX10 Star",
            "-1.8325120054106136983967251435672003010409229121256846741206637961290711916444370587664946204826399529822254414117326686381628426967994560905771081841576459206235088355786921777886193733410526177",
            "1.3232931118299183832032181141553282845026086224092794791567188589580690219080178391114940094095383479900584165976802013239979323432793148307905892292615474713346914559231067621628314952500777586e-20",
            175.7529,
            MDZ_GALLERY_URL,
            "xx10-star.kfr",
        ),
        _pair(
            "bf04-fourfold-boot",
            "BF04 Fourfold Boot",
            "-1.7493148452000870606588924590483767181017388027451266863950258535212250012854329786320591573860444837168084801868424316018028993310027768168714778298029106130651506506558163544586233162324477882",
            "1.4076871905345744769264106927453427017617958116233313682764801278836602396999046278011280927339231029309784356049955346904116649524745984325155005952857117042610872611137857676169502746104489183e-4",
            182.7210,
            MDZ_GALLERY_URL,
            "bf04-4fold-boot-sq.kfr",
        ),
        _pair(
            "bf05-bat",
            "BF05 Bat",
            "-1.7493148452000870606588924590483767181017388027451266863950258535212250012854329786320591573860444837168084801868424316018028993310027768168714778298029106130651506506558163544586233162400456576",
            "1.4076871905345744769264106927453427017617958116233313682764801278836602396999046278011280927339231029309784356049955346904116649524745984325155005952857117042610872611137857676169498533685734167e-4",
            191.3979,
            MDZ_GALLERY_URL,
            "bf05-bat.kfr",
        ),
        _pair(
            "shark-view-11",
            "FractalShark View 11",
            "-1.7691083304074772808923062435577784835600639118623400650778788340418929715985908084423460199088682020437579853971174404281518620786746095361341584243275794459276782934933581123334452910141425659321186244959701232048444218",
            "-0.0090206880570726176003609359849476201123055846741238668897279198171291100002730703257346527485772604116419772842679534355866791857480625214521856184017205123759282792828400173254614353959003473338316788673727096083739377",
            712.6076,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 11",
        ),
        _pair(
            "shark-view-12",
            "FractalShark View 12",
            "-0.1973084884010900400636949584924437797532200156733497896086621693684262735986408266377627689409572069980366084982637715023667826686916289574727768203878915350084840302002338982933530918277669401478364960465526231129559637",
            "1.1038012946026287867175340537321106754714179377909829608698111892311828116339593340641577619320630631272359545249104602024464660368108622972257186686981113108472441866359913631419173375708534023908956818662781845243135850",
            6710.0,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 12",
        ),
        _pair(
            "shark-view-13",
            "FractalShark View 13",
            "-0.1981116002196734448075522850987945354377282414708754357520928216446040579666185891192046341471335540473067233993445865021231667918522445746450594791995187704475257438270456482482062152793444840401649336432054722930651961",
            "-1.0995133542657857115808634031346189677870097136303572938715549689295389425632581503138812246977639087726709908677423161184335187629274263873188431553184305117753521329343866123179772001228771002107363550330517703579998141",
            4406.0,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 13",
        ),
        _pair(
            "shark-view-19",
            "FractalShark View 19",
            "-0.48065550796374945612193350910587992881196972760301218859906076861832853774308262389069947355595792530735710085477971769115391712081609017535787686457114048950617761059125394344431338085233034295",
            "0.637475590124970805209411919505612080979979005679966143690973242042482999140689879878628060058864351901269247080843976263520017910967519862151146952991098489339773644882278954814379692136547821545",
            158.0,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 19",
            screened_log10_zoom=157.0,
        ),
        _pair(
            "shark-view-21",
            "FractalShark View 21",
            "-0.1653530544258343270683475500553350921200942410000263804159124814788460489968512646285743224235602666905877538603965508756281961056039400314471245072847961346455823800680946476070902977915409904979885417319773551801905208",
            "1.0442770648978088053829721335149700336366697520295924336066211790113244608391544077600834019755870845144856659440781374128064070028659954354483203013798531714310185041645017918829379159597014603086807898335389400176151092",
            10871.0,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 21",
        ),
        _pair(
            "shark-view-32",
            "FractalShark View 32",
            "-0.2281554936539618192145720140991260067373983511745080323795979810844156630360130667574210427119704372689395853868164699053606217683389294942385453905473536385036091514212085247889149552407182946498073999359998588167613691",
            "1.1151425080399373597457646363150140681887780904679546036274680334810409092946355168572572006874765228872100322451033324674089490817457188452599842530139888003789611498451538187135582267447926710194039382369014758821696953",
            244240.0,
            FRACTALSHARK_SOURCE_URL,
            "FractalViewPresets.cpp view 32",
        ),
    )
    for point in pair
) + (
    DeepZoomPoint(
        "shark-view-7",
        "FractalShark View 7",
        "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030252478540752851056038295730755908172619776454430617691879015535410340261954919",
        "0.001117567238896768611945287793650368042097805694309796191913683651017675842342387390060146420308670825848799800846008910296521948940339832985255570312674864622603498123149697979818597908682853405147797510782833941585",
        160.993,
        FRACTALSHARK_SOURCE_URL,
        "FractalViewPresets.cpp view 7",
        160.0,
    ),
    DeepZoomPoint(
        "shark-view-8",
        "FractalShark View 8",
        "-1.6225530545095544093937832714855193369815166490586925235310445917701797841889161669038013631146956964774653525559715287987054482808403025247854075285105603829572963367477346607819868788618326566279105639888311967675411750",
        "0.0011175672388967686119452877936503680420978056943097961919136836510176758423423873900601464203086708258487998008460089102965219489403398329852555703126748646226118355222840756725438548472552647319463823283712140095265617",
        668.098,
        FRACTALSHARK_SOURCE_URL,
        "FractalViewPresets.cpp view 8",
        161.0,
    ),
    DeepZoomPoint(
        "shark-view-15",
        "FractalShark View 15",
        "-1.2552386060808794544705762073214718782313298977374713834683672679219039239148526283454222176533392628121047872208077269716197797093371025472466668952255134966668370658175279054871745549423277809914665724804837101524717571495723874748859332474368687432650230295544084499397040284245694572489",
        "0.382138678287792026292480558809852384550778262515404869183027975019781526668183948266451891307973566087869034733711348676099009121244901147848338937814269399000905149775558707603837258390604413630021227035215627036116230455476791069611273040138691411817207284372551905584257155920467249308635",
        260.826,
        FRACTALSHARK_SOURCE_URL,
        "FractalViewPresets.cpp view 15",
        260.0,
    ),
    DeepZoomPoint(
        "shark-view-16",
        "FractalShark View 16",
        "-0.2281792076991057188791220757070906154801563373848351036490006446241507468147564988356371628826904426023128505132018554501675433482727741549694771224847955228931798289163815438743631000392333635842525042153582202499655604449475066164586596459152105106064111624441822700788132655777068944695173876856445256369203065186631701760525784081736244",
        "1.115156767115551104371509493405146139515685158158631166093965276070475149166424130419686602805948430925491857449965291251362671124105939610231137732602258574689237558180672414305416362841285284232472179206525520835146101831409246792045496633731870859449092417056425258013586398208085502164519751503113713282571658615520780020827777231498937",
        301.146,
        FRACTALSHARK_SOURCE_URL,
        "FractalViewPresets.cpp view 16",
        301.0,
    ),
)


DEEP_ZOOM_POINTS_BY_SLUG = {point.slug: point for point in DEEP_ZOOM_POINTS}


# These catalogues deliberately stay separate.  A coordinate in the
# Mandelbrot parameter plane is not automatically meaningful in a Julia,
# Burning Ship, Tricorn, or Multibrot view.  The alternate entries are
# formula-specific starter views; unlike the Mandelbrot catalogue, they are
# not claims of a native e150-safe path.
BURNING_SHIP_SOURCE_URL = "https://www.mathr.co.uk/web/a-burning-ship-fractal-zoom.html"
BURNING_SHIP_EXPLORER_URL = "https://robotmoon.com/burning-ship-fractal/"
TRICORN_SOURCE_URL = "https://paulbourke.net/fractals/tricorn/"
JULIA_SOURCE_URL = "https://khutchins.com/fractals/julia_explorer.html"
JULIA_GALLERY_URL = "https://www.geogebra.org/m/xxchgex7"


def _formula_point(
    slug: str,
    name: str,
    formula: str,
    x: str,
    y: str,
    recommended_log10_zoom: float,
    source_url: str,
    source_name: str,
    *,
    julia_c: tuple[str, str] | None = None,
) -> DeepZoomPoint:
    return DeepZoomPoint(
        slug,
        name,
        x,
        y,
        recommended_log10_zoom,
        source_url,
        source_name,
        recommended_log10_zoom,
        None,
        formula,
        julia_c,
    )


# The Burning Ship centre below is the exact long-decimal centre published
# with Mathr's Burning Ship zoom.  The remaining starter views are centres of
# the mast, small ships, and real-axis structures shown by the explorer at
# ordinary magnifications.
BURNING_SHIP_POINTS = (
    _formula_point(
        "burning-ship-helm",
        "Burning Ship Helm",
        "burning-ship",
        "-1.780425137323272787815501970866799220117789164950755824065998180804054216718401083082315480682414200942106597271163834162257864725732897228477210166344966847497073401e+00",
        "-6.500377933325755658690166726343568470341357651166253908409608154945176557553463044202701111297831544836546730208444477432975243227800009643123524287621756484090422114572084021112115151e-02",
        166.7,
        BURNING_SHIP_SOURCE_URL,
        "Mathr Burning Ship zoom",
    ),
    _formula_point(
        "burning-ship-mast",
        "Largest Ship Mast",
        "burning-ship",
        "-1.77375",
        "-0.05825",
        2.2,
        BURNING_SHIP_EXPLORER_URL,
        "Robot Moon Burning Ship explorer",
    ),
    _formula_point(
        "burning-ship-expanse",
        "Real-axis Expanse",
        "burning-ship",
        "-1.51003475",
        "-0.001153875",
        2.2,
        BURNING_SHIP_EXPLORER_URL,
        "Robot Moon Burning Ship explorer",
    ),
    _formula_point(
        "burning-ship-small-ship",
        "Small Ship near -1.57",
        "burning-ship",
        "-1.56525",
        "-0.017",
        2.0,
        BURNING_SHIP_EXPLORER_URL,
        "Robot Moon Burning Ship explorer",
    ),
    _formula_point(
        "burning-ship-left-edge",
        "Left Edge near -2",
        "burning-ship",
        "-1.98345",
        "0.0",
        1.0,
        BURNING_SHIP_EXPLORER_URL,
        "Robot Moon Burning Ship explorer",
    ),
)


TRICORN_POINTS = (
    _formula_point(
        "tricorn-west-branch",
        "West Branch",
        "tricorn",
        "-1.9412",
        "0.0",
        2.48,
        TRICORN_SOURCE_URL,
        "Paul Bourke Tricorn gallery",
    ),
    _formula_point(
        "tricorn-angled-branch",
        "Angled Branch",
        "tricorn",
        "-0.85",
        "0.145",
        1.48,
        TRICORN_SOURCE_URL,
        "Paul Bourke Tricorn gallery",
    ),
    _formula_point(
        "tricorn-main-branch",
        "Main Branch",
        "tricorn",
        "-1.25",
        "0.0",
        0.60,
        TRICORN_SOURCE_URL,
        "Paul Bourke Tricorn gallery",
    ),
    _formula_point(
        "tricorn-centre",
        "Centre",
        "tricorn",
        "-0.4",
        "0.0",
        0.0,
        TRICORN_SOURCE_URL,
        "Paul Bourke Tricorn gallery",
    ),
)


MULTIBROT3_POINTS = (
    _formula_point(
        "multibrot3-lobe",
        "Lower-right Lobe",
        "multibrot3",
        "0.8",
        "-0.8",
        0.60,
        TRICORN_SOURCE_URL,
        "Paul Bourke Multibrot gallery",
    ),
    _formula_point(
        "multibrot3-centre",
        "Centre",
        "multibrot3",
        "0.0",
        "0.0",
        0.0,
        TRICORN_SOURCE_URL,
        "Paul Bourke Multibrot gallery",
    ),
    _formula_point(
        "multibrot3-upper-left",
        "Upper-left Lobe",
        "multibrot3",
        "-0.8",
        "0.8",
        0.60,
        TRICORN_SOURCE_URL,
        "Paul Bourke Multibrot gallery",
    ),
)


# For Julia sets, c is part of the preset.  The viewport centre starts at the
# origin because that is the natural whole-set view; users can still type any
# exact REAL,IMAG centre in the GUI or CLI.
JULIA_POINTS = (
    _formula_point(
        "julia-dragon",
        "Dragon / Connected",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_SOURCE_URL,
        "Julia Set Explorer",
        julia_c=("-0.8", "0.156"),
    ),
    _formula_point(
        "julia-douady-rabbit",
        "Douady Rabbit",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_GALLERY_URL,
        "GeoGebra Super zoom Julia",
        julia_c=("-0.123", "0.745"),
    ),
    _formula_point(
        "julia-dendrite",
        "Dendrite",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_GALLERY_URL,
        "GeoGebra Super zoom Julia",
        julia_c=("0.0", "1.0"),
    ),
    _formula_point(
        "julia-disconnected",
        "Disconnected Dust",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_GALLERY_URL,
        "GeoGebra Super zoom Julia",
        julia_c=("0.285", "-0.01"),
    ),
    _formula_point(
        "julia-seahorse",
        "Seahorse",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_SOURCE_URL,
        "Julia Set Explorer",
        julia_c=("-0.4", "0.6"),
    ),
    _formula_point(
        "julia-basilica",
        "Basilica",
        "julia",
        "0.0",
        "0.0",
        1.0,
        JULIA_SOURCE_URL,
        "Julia Set Explorer",
        julia_c=("-1.0", "0.0"),
    ),
)


FORMULA_POINT_CATALOGUES = {
    "mandelbrot": DEEP_ZOOM_POINTS,
    "julia": JULIA_POINTS,
    "burning-ship": BURNING_SHIP_POINTS,
    "tricorn": TRICORN_POINTS,
    "multibrot3": MULTIBROT3_POINTS,
}
FORMULA_POINTS_BY_SLUG = {
    formula: {point.slug: point for point in points}
    for formula, points in FORMULA_POINT_CATALOGUES.items()
}
ALL_FRACTAL_POINTS = tuple(
    point for points in FORMULA_POINT_CATALOGUES.values() for point in points
)
