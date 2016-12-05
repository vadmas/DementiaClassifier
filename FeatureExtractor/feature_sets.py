def get_fraser_top_50():
    return [
        # POS FEATURES
        "RatioPronoun",
        "NP_to_PRP",
        # "",
        "NumAdverbs",
        "ADVP_to_RB",
        "VP_to_VBG_PP",
        # "VP_to_IN_S", Need to locate this feature
        "VP_to_AUX_ADJP",
        "VP_to_AUX_VP",
        "VP_to_VBG",
        "VP_to_AUX",
        "VP_to_VBD_NP",
        "INTJ_to_UH",
        "NP_to_DT_NN",
        "proportion_below_threshold_0.5",
        "NumVerbs",
        "NumNouns",
        "MeanWordLength",
        "HonoreStatistic",
        "NumInflectedVerbs",
        "avg_cos_dist",
        "VPTypeRate",
        "PProportion",
        "PPTypeRate",
        # InfoUnits
        "keywordIUObjectWindow",
        "binaryIUObjectWindow",
        "keywordIUObjectSink",
        "binaryIUObjectSink",
        "keywordIUObjectCookie",
        "binaryIUObjectCookie",
        "keywordIUObjectCurtains",
        "binaryIUObjectCurtains",
        "binaryIUSubjectGirl",
        "binaryIUObjectDishes",
        "keywordIUObjectStool",
        "binaryIUObjectStool",
        "keywordIUSubjectWoman",
        "binaryIUSubjectWoman",
        # PsychoLing
        "getFamiliarityScore",
        "getConcretenessScore",
        "getImagabilityScore",
        "getAoaScore",
        "getSUBTLWordScores",
        "getLightVerbCount",
        # Acoustic
        "mfcc1_skewness",
        "mfcc2_skewness",
        "mfcc8_skewness",
        "mfcc12_skewness",
        "mfcc5_kurtosis",
        "mfcc3_vel_kurtosis"
    ]


def get_cfg_features():
    return [
        "ADVP_to_RB",
        "INTJ_to_UH",
        "NP_to_DT_NN",
        "NP_to_PRP",
        "ROOT_to_FRAG",
        "VP_to_AUX",
        "VP_to_AUX_ADJP",
        "VP_to_AUX_VP",
        "VP_to_VBD_NP",
        "VP_to_VBG",
        "VP_to_VBG_PP",
        "C",
        "CN",
        "CN_C",
        "CN_T",
        "CONJP",
        "CP",
        "CP_C",
        "CP_T",
        "CT",
        "CT_T",
        "C_S",
        "C_T",
        "DC",
        "DC_C",
        "DC_T",
        "S",
        "T",
        "TTR",
        "T_S",
        "UCP",
        "VP",
        "VP_T",
        "AvgNPTypeLengthEmbedded",
        "AvgNPTypeLengthNonEmbedded",
        "AvgPPTypeLengthEmbedded",
        "AvgPPTypeLengthNonEmbedded",
        "AvgVPTypeLengthEmbedded",
        "AvgVPTypeLengthNonEmbedded",
        "W",
        "WHADJP",
        "WHAVP",
        "WHNP",
        "WHPP",
        "X",
        "FRAG",
        "INTJ",
        "LST",
        "NPTypeRate",
        "VPTypeRate",
        "PPTypeRate",
        "PProportion",
        "NPProportion",
        "VPProportion",
        "NAC",
        "NP",
        "NX",
        "PP",
        "PRN",
        "PRT",
        "QP",
        "RRC",
        "ADJP",
        "ADVP",
    ]


def get_syntactic_complexity_features():
    return[
        "MeanLengthOfSentence",
        "MeanWordLength",
        "MLC",
        "MLS",
        "MLT",
        "DisfluencyFrequency",
        "TotalNumberOfWords",
        "number_of_utterances",
        "tree_height"
    ]


def get_psycholinguistic_features():
    return[
        "getAoaScore",
        "getConcretenessScore",
        "getFamiliarityScore",
        "getImagabilityScore",
        "getLightVerbCount",
        "getSUBTLWordScores",
    ]


def get_vocabulary_richness_features():
    return [
        "MATTR",
        "BrunetIndex",
        "HonoreStatistic",
    ]


def get_repetitiveness_features():
    return [
        "min_cos_dist",
        "proportion_below_threshold_0",
        "proportion_below_threshold_0.3",
        "proportion_below_threshold_0.5",
        "avg_cos_dist"
    ]


def get_acoustics_features():
    return [
        "mfcc13_mean",
        "mfcc1_mean",
        "mfcc7_vel_var",
        "mfcc1_vel_mean",
        "mfcc2_accel_skewness",
        "mfcc4_accel_skewness",
        "mfcc4_vel_kurtosis",
        "mfcc6_vel_skewness",
        "mfcc11_vel_skewness",
        "mfcc4_kurtosis",
        "mfcc8_vel_kurtosis",
        "mfcc7_vel_kurtosis",
        "mfcc3_vel_var",
        "mfcc3_accel_kurtosis",
        "mfcc8_skewness",
        "mfcc4_var",
        "mfcc8_accel_kurtosis",
        "mfcc10_kurtosis",
        "mfcc4_accel_kurtosis",
        "mfcc5_accel_skewness",
        "mfcc9_var",
        "mfcc3_accel_var",
        "mfcc7_vel_mean",
        "mfcc12_vel_var",
        "mfcc8_kurtosis",
        "mfcc7_mean",
        "mfcc7_accel_mean",
        "mfcc2_accel_mean",
        "mfcc1_accel_skewness",
        "mfcc2_var",
        "mfcc5_kurtosis",
        "mfcc11_accel_var",
        "mfcc12_skewness",
        "mfcc9_mean",
        "mfcc12_vel_mean",
        "mfcc2_vel_skewness",
        "mfcc12_kurtosis",
        "mfcc4_vel_skewness",
        "mfcc10_skewness",
        "mfcc4_vel_var",
        "mfcc12_mean",
        "mfcc6_vel_mean",
        "mfcc10_vel_kurtosis",
        "mfcc1_accel_mean",
        "mfcc1_kurtosis",
        "mfcc7_var",
        "mfcc10_mean",
        "mfcc7_kurtosis",
        "mfcc5_vel_mean",
        "mfcc2_mean",
        "mfcc7_accel_kurtosis",
        "mfcc2_accel_kurtosis",
        "mfcc6_vel_kurtosis",
        "mfcc1_vel_kurtosis",
        "mfcc11_skewness",
        "mfcc6_mean",
        "mfcc8_vel_var",
        "mfcc3_skewness",
        "mfcc5_skewness",
        "mfcc2_accel_var",
        "mfcc6_accel_skewness",
        "mfcc5_accel_mean",
        "mfcc1_vel_var",
        "mfcc6_vel_var",
        "mfcc10_accel_kurtosis",
        "mfcc8_var",
        "mfcc11_vel_kurtosis",
        "mfcc10_vel_mean",
        "mfcc1_skewness",
        "mfcc1_accel_var",
        "mfcc7_vel_skewness",
        "mfcc9_vel_mean",
        "mfcc2_vel_mean",
        "mfcc7_accel_var",
        "mfcc13_vel_var",
        "mfcc3_mean",
        "mfcc1_var",
        "mfcc_skewness",
        "mfcc5_accel_kurtosis",
        "mfcc1_vel_skewness",
        "mfcc5_accel_var",
        "mfcc12_vel_kurtosis",
        "mfcc6_accel_kurtosis",
        "mfcc5_mean",
        "mfcc12_accel_kurtosis",
        "mfcc4_mean",
        "mfcc11_kurtosis",
        "mfcc6_skewness",
        "mfcc2_vel_var",
        "mfcc9_skewness",
        "mfcc12_accel_skewness",
        "mfcc12_accel_mean",
        "mfcc8_accel_skewness",
        "mfcc8_vel_skewness",
        "mfcc13_accel_var",
        "mfcc5_vel_var",
        "mfcc6_var",
        "mfcc13_kurtosis",
        "mfcc11_mean",
        "mfcc12_accel_var",
        "mfcc10_vel_var",
        "mfcc8_vel_mean",
        "mfcc8_mean",
        "mfcc10_accel_skewness",
        "mfcc6_accel_var",
        "mfcc12_vel_skewness",
        "mfcc4_accel_var",
        "mfcc13_vel_kurtosis",
        "mfcc11_accel_kurtosis",
        "mfcc10_accel_var",
        "mfcc9_kurtosis",
        "mfcc3_vel_mean",
        "mfcc11_vel_var",
        "mfcc13_skewness",
        "mfcc9_accel_skewness",
        "mfcc6_accel_mean",
        "mfcc5_vel_kurtosis",
        "mfcc4_accel_mean",
        "mfcc3_accel_mean",
        "mfcc10_accel_mean",
        "mfcc9_accel_var",
        "mfcc3_var",
        "mfcc13_vel_mean",
        "mfcc13_var",
        "mfcc2_kurtosis",
        "mfcc9_accel_mean",
        "mfcc6_kurtosis",
        "mfcc9_vel_skewness",
        "mfcc13_accel_mean",
        "mfcc5_var",
        "mfcc13_accel_kurtosis",
        "mfcc11_accel_skewness",
        "mfcc13_accel_skewness",
        "mfcc4_skewness",
        "mfcc2_vel_kurtosis",
        "mfcc9_accel_kurtosis",
        "mfcc11_var",
        "mfcc7_skewness",
        "mfcc11_vel_mean",
        "mfcc3_accel_skewness",
        "mfcc_kurtosis",
        "mfcc7_accel_skewness",
        "mfcc3_vel_kurtosis",
        "mfcc2_skewness",
        "mfcc11_accel_mean",
        "mfcc9_vel_kurtosis",
        "mfcc4_vel_mean",
        "mfcc3_kurtosis",
        "mfcc9_vel_var",
        "mfcc8_accel_mean",
        "mfcc12_var",
        "mfcc5_vel_skewness",
        "mfcc13_vel_skewness",
        "mfcc10_vel_skewness",
        "mfcc3_vel_skewness",
        "mfcc1_accel_kurtosis",
        "mfcc10_var",
        "mfcc8_accel_var",
        "energy_accel_mean",
        "energy_vel_mean",
        "energy_accel_var",
        "energy_skewness",
        "energy_vel_skewness",
        "energy_kurtosis",
        "energy_vel_kurtosis",
        "energy_mean",
        "energy_var",
        "energy_vel_var",
        "ff_var",
        "energy_accel_skewness",
        "energy_accel_kurtosis",
        "ff_mean"
    ]


def get_discourse_features():
    return [
        "Comparison",
        "edu_rate",
        "Topic-Change",
        "Summary",
        "Topic-Comment",
        "Same-Unit",
        "Evaluation",
        "Contrast",
        "Elaboration",
        "Attribution",
        "TextualOrganization",
        "Cause",
        "Explanation",
        "Enablement",
        "Joint",
        "depth",
        "Background",
        "Temporal",
        "Condition",
        "Manner-Means"
    ]


def get_parts_of_speech_features():
    return [
        "NumNouns",
        "NumVerbs",
        "NumberOfNID",
        "MeanWordLength",
        "NumAdverbs",
        "NumAdjectives",
        "NumDeterminers",
        "NumInterjections",
        "NumInflectedVerbs",
        "NumCoordinateConjunctions",
        "NumSubordinateConjunctions",
        "RatioNoun",
        "RatioVerb",
        "RatioPronoun",
        "RatioCoordinate"
    ]


def get_information_content_features():
    return [
        "keywordIUObjectCookie",
        "keywordIUObjectCupboard",
        "keywordIUObjectCurtains",
        "keywordIUObjectDishcloth",
        "keywordIUObjectDishes",
        "keywordIUObjectJar",
        "keywordIUObjectPlate",
        "keywordIUObjectSink",
        "keywordIUObjectStool",
        "keywordIUObjectWater",
        "keywordIUObjectWindow",
        "keywordIUPlaceExterior",
        "keywordIUPlaceKitchen",
        "keywordIUSubjectBoy",
        "keywordIUSubjectGirl",
        "keywordIUSubjectWoman",
        "binaryIUActionBoyTaking",
        "binaryIUActionStoolFalling",
        "binaryIUActionWaterOverflowing",
        "binaryIUActionWomanDryingWashing",
        "binaryIUObjectCookie",
        "binaryIUObjectCupboard",
        "binaryIUObjectCurtains",
        "binaryIUObjectDishcloth",
        "binaryIUObjectDishes",
        "binaryIUObjectJar",
        "binaryIUObjectPlate",
        "binaryIUObjectSink",
        "binaryIUObjectStool",
        "binaryIUObjectWater",
        "binaryIUObjectWindow",
        "binaryIUPlaceExterior",
        "binaryIUPlaceKitchen",
        "binaryIUSubjectBoy",
        "binaryIUSubjectGirl",
        "binaryIUSubjectWoman"
    ]


def get_demographic_features():
    return [
        "age",
        "gender_int",
    ]


def get_general_keyword_features():
    return [
        'count_of_general_keyword',
        'general_keyword_to_non_keyword_ratio',
        'general_keyword_type_to_token_ratio',
        'percentage_of_general_keywords_mentioned'
    ]


def get_leftside_keyword_features():
    return [
        'count_of_leftside_keyword', 
        'leftside_keyword_to_non_keyword_ratio',
        'leftside_keyword_type_to_token_ratio',
        'percentage_of_leftside_keywords_mentioned',
    ]


def get_rightside_keyword_features():
    return [
        'count_of_rightside_keyword',
        'percentage_of_rightside_keywords_mentioned',
        'rightside_keyword_to_non_keyword_ratio',
        'rightside_keyword_type_to_token_ratio',
    ]



def get_leftside_rightside_features():
     return [
        'count_of_leftside_keyword', 
        'leftside_keyword_to_non_keyword_ratio',
        'leftside_keyword_type_to_token_ratio',
        'percentage_of_leftside_keywords_mentioned',
        'count_of_rightside_keyword',
        'percentage_of_rightside_keywords_mentioned',
        'rightside_keyword_to_non_keyword_ratio',
        'rightside_keyword_type_to_token_ratio',
        "count_ls_rs_switches",
    ]

def get_ls_rs_polynomial():
    return[
        'squared_ls_count',
        'squared_ls_kw_to_w_ratio',
        'squared_ls_ty_to_tok_ratio',
        'squared_prcnt_ls_uttered',
        'squared_rs_count',
        'squared_prcnt_rs_uttered',
        'squared_rs_kw_to_w_ratio',
        'squared_rs_ty_to_tok_ratio',
        'squared_count_ls_rs_switches',
        'interaction_ls_count_rs_count',
        'interaction_ls_count_prcnt_rs_uttered',
        'interaction_ls_count_rs_kw_to_w_ratio',
        'interaction_ls_count_rs_ty_to_tok_ratio',
        'interaction_ls_kw_to_w_ratio_rs_count',
        'interaction_ls_kw_to_w_ratio_prcnt_rs_uttered',
        'interaction_ls_kw_to_w_ratio_rs_kw_to_w_ratio',
        'interaction_ls_kw_to_w_ratio_rs_ty_to_tok_ratio',
        'interaction_ls_ty_to_tok_ratio_rs_count',
        'interaction_ls_ty_to_tok_ratio_prcnt_rs_uttered',
        'interaction_ls_ty_to_tok_ratio_rs_kw_to_w_ratio',
        'interaction_ls_ty_to_tok_ratio_rs_ty_to_tok_ratio',
        'interaction_prcnt_ls_uttered_rs_count',
        'interaction_prcnt_ls_uttered_prcnt_rs_uttered',
        'interaction_prcnt_ls_uttered_rs_kw_to_w_ratio',
        'interaction_prcnt_ls_uttered_rs_ty_to_tok_ratio',
        'interaction_count_ls_rs_switches_ls_count',
        'interaction_count_ls_rs_switches_ls_kw_to_w_ratio',
        'interaction_count_ls_rs_switches_ls_ty_to_tok_ratio',
        'interaction_count_ls_rs_switches_prcnt_ls_uttered',
        'interaction_count_ls_rs_switches_rs_count',
        'interaction_count_ls_rs_switches_prcnt_rs_uttered',
        'interaction_count_ls_rs_switches_rs_kw_to_w_ratio',
        'interaction_count_ls_rs_switches_rs_ty_to_tok_ratio'
]


def get_all_groups():
    groups = {}
    groups["cfg"]                  = get_cfg_features()
    groups["syntactic_complexity"] = get_syntactic_complexity_features()
    groups["psycholinguistic"]     = get_psycholinguistic_features()
    groups["vocabulary_richness"]  = get_vocabulary_richness_features()
    groups["repetitiveness"]       = get_repetitiveness_features()
    groups["acoustics"]            = get_acoustics_features()
    groups["discourse"]            = get_discourse_features()
    groups["parts_of_speech"]      = get_parts_of_speech_features()
    groups["demographic"]          = get_demographic_features()
    groups["information_content"]  = get_information_content_features()
    groups['get_general_keyword_features'] = get_general_keyword_features()
    groups['get_leftside_rightside_features'] = get_leftside_rightside_features()
    return groups
