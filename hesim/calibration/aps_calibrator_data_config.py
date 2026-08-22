from os.path import dirname, join

eiger_dark_exp_dirs = {
    "1": join(dirname(__file__), "../../calibration_data/Examples/black_level_calibration-1ms"),
    "5": join(dirname(__file__), "../../calibration_data/Examples/black_level_calibration-5ms"),
    "50": join(dirname(__file__), "../../calibration_data/Examples/black_level_calibration-50ms"),
    "80": join(dirname(__file__), "../../calibration_data/Examples/black_level_calibration-80ms"),
}
eiger_dynamic_fodler = "../../calibration_data/ARGB_ERGB_Eiger/"
eiger_dynamic_exp_dirs = {
    "1": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp01ms_20240516153213128/APS/quadbayer_bit8_3264_2448_20240516153213128",
    ),
    "2": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp02ms_20240516153221655/APS/quadbayer_bit8_3264_2448_20240516153221655",
    ),
    "5": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp05ms_20240516153235224/APS/quadbayer_bit8_3264_2448_20240516153235224",
    ),
    "10": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp10ms_20240516153244340/APS/quadbayer_bit8_3264_2448_20240516153244340",
    ),
    "20": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp20ms_20240516153253743/APS/quadbayer_bit8_3264_2448_20240516153253743",
    ),
    "40": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp40ms_20240516153312041/APS/quadbayer_bit8_3264_2448_20240516153312041",
    ),
    "50": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp50ms_20240516153331327/APS/quadbayer_bit8_3264_2448_20240516153331327",
    ),
    "80": join(
        dirname(__file__),
        eiger_dynamic_fodler,
        "resolution_board_color_checker_exp80ms_20240516153343982/APS/quadbayer_bit8_3264_2448_20240516153343982",
    ),
}

sensitivity7_folder = "./calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/"
gen2_dark_exp_dirs = {
    "1": join(
        sensitivity7_folder,
        "20250903_dark_1_7/APS/quadbayer_10bit_3264_2448_20250903223935090/aps_raw",
    ),
    "5": join(
        sensitivity7_folder,
        "20250904_dark_5_7/APS/quadbayer_10bit_3264_2448_20250904231531252/aps_raw",
    ),
    "20": join(
        sensitivity7_folder,
        "20250904_dark_20_7/APS/quadbayer_10bit_3264_2448_20250904231920768/aps_raw",
    ),
    "50": join(
        sensitivity7_folder,
        "20250904_dark_50_7/APS/quadbayer_10bit_3264_2448_20250904232324680/aps_raw",
    ),
    "80": join(
        sensitivity7_folder,
        "20250904_dark_80_7/APS/quadbayer_10bit_3264_2448_20250904232701058/aps_raw",
    ),
}

gen2_dynamic_exp_dirs = {
    "1": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922170640101sen7_1/APS/quadbayer_10bit_3264_2448_20250922170640101/aps_raw",
    "2": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922170514149sen7_2/APS/quadbayer_10bit_3264_2448_20250922170514149/aps_raw",
    "5": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922170346721sen7_5/APS/quadbayer_10bit_3264_2448_20250922170346721/aps_raw",
    "10": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922170220537sen7_10/APS/quadbayer_10bit_3264_2448_20250922170220537/aps_raw",
    "20": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922170058811sen7_20/APS/quadbayer_10bit_3264_2448_20250922170058811/aps_raw",
    "40": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922165939212sen7_40/APS/quadbayer_10bit_3264_2448_20250922165939212/aps_raw",
    "50": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922165807108sen7_50/APS/quadbayer_10bit_3264_2448_20250922165807108/aps_raw",
    "80": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922165651933sen7_80/APS/quadbayer_10bit_3264_2448_20250922165651933/aps_raw",
}

gen2_dynamic_exp_dirs_board = {
    "10": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250923163255026_sen7_10/APS/quadbayer_10bit_3264_2448_20250923163255026/aps_raw",
    "20": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250923163506082_sen7_20/APS/quadbayer_10bit_3264_2448_20250923163506082/aps_raw",
    "30": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250923163801942_sen7_30/APS/quadbayer_10bit_3264_2448_20250923163801942/aps_raw",
    "80": "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/20250922165651933sen7_80/APS/quadbayer_10bit_3264_2448_20250922165651933/aps_raw",
}
