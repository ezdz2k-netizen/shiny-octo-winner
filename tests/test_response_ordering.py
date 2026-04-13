import unittest
import shutil
from pathlib import Path

import pandas as pd

from app.main import (
    _build_question_mappings_from_codebook,
    _build_question_mappings,
    detect_concept_groups,
    _normalize_response_value,
    _mr_display_metadata,
    _transpose_result,
    _resolve_uploaded_workbook_roles,
    load_mapping,
    detect_mr_groups,
    tabulate_mr,
    tabulate_sr_concept_comparison,
    tabulate_sr,
)


class ResponseOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text_df = pd.DataFrame(
            {
                "Q1": ["Agree", "Neutral", "Agree", "Disagree"],
                "Banner": ["Male", "Female", "Female", "Male"],
            }
        )
        self.value_df = pd.DataFrame(
            {
                "Q1": ["2", 3, "2", 1],
                "Banner": [10, 20, 20, 10],
            }
        )
        self.bundle = _build_question_mappings(self.text_df, self.value_df)

    def test_normalizer_aligns_numeric_strings_and_integers(self) -> None:
        self.assertEqual(_normalize_response_value("1"), "1")
        self.assertEqual(_normalize_response_value(1), "1")
        self.assertEqual(_normalize_response_value("1.0"), "1")
        self.assertIsNone(_normalize_response_value("   "))

    def test_mapping_uses_paired_labels(self) -> None:
        q1 = self.bundle.questions["Q1"]
        self.assertEqual(q1.by_code["2"].label, "Agree")
        self.assertEqual(q1.by_code["3"].label, "Neutral")
        self.assertEqual(q1.by_code["1"].label, "Disagree")

    def test_default_row_order_is_numeric_ascending(self) -> None:
        result = tabulate_sr(
            self.value_df,
            bundle=self.bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=True,
            include_base=True,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["Base", "Disagree", "Agree", "Neutral"])

    def test_missing_code_detection_is_explicit(self) -> None:
        bad_df = self.value_df.copy()
        bad_df.loc[0, "Q1"] = 999
        with self.assertRaisesRegex(RuntimeError, "missing from Value.xlsx"):
            tabulate_sr(
                bad_df,
                bundle=self.bundle,
                row_var="Q1",
                col_var="Banner",
                col_var2=None,
                include_totals=False,
                include_base=False,
                out_counts=True,
                out_rowpct=False,
                out_colpct=False,
            )

    def test_zero_count_categories_are_preserved(self) -> None:
        subset_df = self.value_df.iloc[:3].copy()
        result = tabulate_sr(
            subset_df,
            bundle=self.bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            include_all_categories=True,
        )
        rows = result["tables"][0]["rows"]
        self.assertEqual(rows[0]["__label__"], "Disagree")
        self.assertEqual(rows[0]["Male"], 0)
        self.assertEqual(rows[0]["Female"], 0)

    def test_show_row_codes_prefixes_output_labels(self) -> None:
        result = tabulate_sr(
            self.value_df,
            bundle=self.bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            show_row_codes=True,
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels[0], "1 - Disagree")

    def test_numeric_code_sort_ascending(self) -> None:
        result = tabulate_sr(
            self.value_df,
            bundle=self.bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            show_row_codes=True,
            row_sort_mode="code_asc",
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["1 - Disagree", "2 - Agree", "3 - Neutral"])

    def test_numeric_code_sort_descending(self) -> None:
        result = tabulate_sr(
            self.value_df,
            bundle=self.bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            show_row_codes=True,
            row_sort_mode="code_desc",
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["3 - Neutral", "2 - Agree", "1 - Disagree"])

    def test_columns_sort_ascending_by_raw_code(self) -> None:
        text_df = pd.DataFrame(
            {
                "Q1": ["Agree", "Neutral", "Disagree"],
                "Banner": ["Sixty", "Twenty", "Forty"],
            }
        )
        value_df = pd.DataFrame(
            {
                "Q1": [2, 3, 1],
                "Banner": [60, 20, 40],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        result = tabulate_sr(
            value_df,
            bundle=bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
        )
        self.assertEqual(result["tables"][0]["columns"], ["Twenty", "Forty", "Sixty"])

    def test_bottom2_box_is_added_for_ascending_sort(self) -> None:
        rich_text_df = pd.DataFrame(
            {
                "Q1": ["One", "Two", "Three", "Four", "Five"],
                "Banner": ["Male", "Male", "Male", "Male", "Male"],
            }
        )
        rich_value_df = pd.DataFrame(
            {
                "Q1": [1, 2, 3, 4, 5],
                "Banner": [10, 10, 10, 10, 10],
            }
        )
        bundle = _build_question_mappings(rich_text_df, rich_value_df)
        result = tabulate_sr(
            rich_value_df,
            bundle=bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            include_bottom2_box=True,
            row_sort_mode="code_asc",
        )
        rows = result["tables"][0]["rows"]
        self.assertEqual(rows[0]["__label__"], "Bottom 2 Box")
        self.assertEqual(rows[0]["Male"], 2)

    def test_top2_box_is_added_for_descending_sort(self) -> None:
        rich_text_df = pd.DataFrame(
            {
                "Q1": ["One", "Two", "Three", "Four", "Five"],
                "Banner": ["Male", "Male", "Male", "Male", "Male"],
            }
        )
        rich_value_df = pd.DataFrame(
            {
                "Q1": [1, 2, 3, 4, 5],
                "Banner": [10, 10, 10, 10, 10],
            }
        )
        bundle = _build_question_mappings(rich_text_df, rich_value_df)
        result = tabulate_sr(
            rich_value_df,
            bundle=bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            include_top2_box=True,
            row_sort_mode="code_desc",
        )
        rows = result["tables"][0]["rows"]
        self.assertEqual(rows[0]["__label__"], "Top 2 Box")
        self.assertEqual(rows[0]["Male"], 2)

    def test_numeric_code_sort_falls_back_for_non_numeric_codes(self) -> None:
        text_df = pd.DataFrame(
            {
                "QAlpha": ["Alpha", "Beta"],
                "Banner": ["Male", "Female"],
            }
        )
        value_df = pd.DataFrame(
            {
                "QAlpha": ["A", "B"],
                "Banner": [10, 20],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        result = tabulate_sr(
            value_df,
            bundle=bundle,
            row_var="QAlpha",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            row_sort_mode="code_asc",
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["Alpha", "Beta"])

    def test_detect_concept_groups_handles_prefixed_concept_columns(self) -> None:
        detection = detect_concept_groups(
            [
                "Concept 1: Q1_1. Purchase intent",
                "Concept 2: Q1_2. Purchase intent",
                "Concept 3: Q1_3. Purchase intent",
                "Gender",
            ]
        )
        self.assertEqual(len(detection["concept_groups"]), 1)
        group = detection["concept_groups"][0]
        self.assertEqual(group["label"], "Q1. Purchase intent")
        self.assertEqual(
            [member["concept_label"] for member in group["members"]],
            ["Concept 1", "Concept 2", "Concept 3"],
        )

    def test_detect_concept_groups_handles_numbered_export_prefixes(self) -> None:
        detection = detect_concept_groups(
            [
                "31 : Concept 1: Q1_1. Purchase intent",
                "40 : Concept 2: Q1_2. Purchase intent",
                "49 : Concept 3: Q1_3. Purchase intent",
                "58 : Concept 4: Q1_4. Purchase intent",
            ]
        )
        self.assertEqual(len(detection["concept_groups"]), 1)
        group = detection["concept_groups"][0]
        self.assertEqual(group["label"], "Q1. Purchase intent")
        self.assertEqual(len(group["members"]), 4)

    def test_tabulate_sr_concept_comparison_reshapes_wide_concepts(self) -> None:
        text_df = pd.DataFrame(
            {
                "Concept 1: Q1_1. Purchase intent": ["Top", "Bottom"],
                "Concept 2: Q1_2. Purchase intent": ["Bottom", "Top"],
                "Concept 3: Q1_3. Purchase intent": ["Top", "Top"],
            }
        )
        value_df = pd.DataFrame(
            {
                "Concept 1: Q1_1. Purchase intent": [1, 2],
                "Concept 2: Q1_2. Purchase intent": [2, 1],
                "Concept 3: Q1_3. Purchase intent": [1, 1],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        detection = detect_concept_groups(list(value_df.columns))
        group = detection["member_to_group"]["Concept 1: Q1_1. Purchase intent"]

        result = tabulate_sr_concept_comparison(
            value_df,
            bundle=bundle,
            concept_group=group,
            top_banner_var=None,
            include_totals=False,
            include_base=True,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
        )

        self.assertEqual(result["row_label"], "Q1. Purchase intent")
        self.assertEqual(result["col_label"], "Concept")
        self.assertEqual(result["tables"][0]["columns"], ["C1", "C2", "C3"])
        rows = result["tables"][0]["rows"]
        self.assertEqual(rows[0]["__label__"], "Base")
        self.assertEqual(rows[0]["C1"], 2)
        self.assertEqual(rows[0]["C2"], 2)
        self.assertEqual(rows[0]["C3"], 2)
        self.assertEqual(rows[1]["__label__"], "Top")
        self.assertEqual(rows[1]["C1"], 1)
        self.assertEqual(rows[1]["C2"], 1)
        self.assertEqual(rows[1]["C3"], 2)
        self.assertEqual(rows[2]["__label__"], "Bottom")
        self.assertEqual(rows[2]["C1"], 1)
        self.assertEqual(rows[2]["C2"], 1)
        self.assertEqual(rows[2]["C3"], 0)

    def test_tabulate_sr_concept_comparison_uses_top_banner_layer(self) -> None:
        text_df = pd.DataFrame(
            {
                "Concept 1: Q1_1. Purchase intent": ["Top", "Bottom"],
                "Concept 2: Q1_2. Purchase intent": ["Bottom", "Top"],
                "Gender": ["Male", "Female"],
            }
        )
        value_df = pd.DataFrame(
            {
                "Concept 1: Q1_1. Purchase intent": [1, 2],
                "Concept 2: Q1_2. Purchase intent": [2, 1],
                "Gender": [10, 20],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        detection = detect_concept_groups(list(value_df.columns))
        group = detection["member_to_group"]["Concept 1: Q1_1. Purchase intent"]

        result = tabulate_sr_concept_comparison(
            value_df,
            bundle=bundle,
            concept_group=group,
            top_banner_var="Gender",
            include_totals=False,
            include_base=True,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
        )

        self.assertEqual(result["col_label"], "Gender | Concept")
        table = result["tables"][0]
        self.assertEqual(table["columns"], ["Male | C1", "Male | C2", "Female | C1", "Female | C2"])
        self.assertTrue(table["header_layout"]["two_layer"])
        self.assertEqual(
            [group["label"] for group in table["header_layout"]["top_groups"]],
            ["Male", "Female"],
        )
        self.assertEqual(table["header_layout"]["bottom_labels"], ["C1", "C2", "C1", "C2"])

    def test_upload_role_detection_handles_swapped_picker_order(self) -> None:
        base = Path("tests") / "_tmp_upload_role_detection"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        try:
            text_path = base / "labels.xlsx"
            value_path = base / "Value.xlsx"
            self.text_df.to_excel(text_path, index=False)
            self.value_df.to_excel(value_path, index=False)

            detected_text, detected_value, bundle = _resolve_uploaded_workbook_roles(
                str(value_path),
                "Value.xlsx",
                str(text_path),
                "labels.xlsx",
            )

            self.assertEqual(Path(detected_text).name, "labels.xlsx")
            self.assertEqual(Path(detected_value).name, "Value.xlsx")
            self.assertEqual(bundle.questions["Q1"].by_code["2"].label, "Agree")
        finally:
            if base.exists():
                shutil.rmtree(base)

    def test_detect_mr_groups_uses_separator_stems(self) -> None:
        df = pd.DataFrame(
            {
                "Q5: Brand awareness | TV": [1, 0, 1],
                "Q5: Brand awareness | Digital": [0, 1, 1],
                "Q5: Brand awareness | Store": [0, 0, 1],
                "Age": [25, 30, 35],
            }
        )
        result = detect_mr_groups(df)
        self.assertEqual(len(result["mr_groups"]), 1)
        group = result["mr_groups"][0]
        self.assertEqual(group["stem"], "Q5 : Brand awareness")
        self.assertEqual(
            group["columns"],
            [
                "Q5: Brand awareness | TV",
                "Q5: Brand awareness | Digital",
                "Q5: Brand awareness | Store",
            ],
        )
        self.assertEqual(group["options"], ["TV", "Digital", "Store"])
        self.assertIn("Age", result["non_mr_columns"])

    def test_detect_mr_groups_rejects_non_binary_matches(self) -> None:
        df = pd.DataFrame(
            {
                "Q6 - Reasons - Price": [1, 0, 1],
                "Q6 - Reasons - Quality": [2, 1, 3],
                "Q6 - Reasons - Scent": [0, 1, 0],
            }
        )
        result = detect_mr_groups(df)
        self.assertEqual(result["mr_groups"], [])

    def test_detect_mr_groups_has_similarity_fallback(self) -> None:
        df = pd.DataFrame(
            {
                "Brand imagery improves mood soft finish": [1, 0, 1],
                "Brand imagery improves mood deep clean": [0, 1, 0],
                "Region": ["East", "West", "East"],
            }
        )
        result = detect_mr_groups(df)
        self.assertEqual(len(result["mr_groups"]), 1)
        group = result["mr_groups"][0]
        self.assertEqual(group["columns"][0], "Brand imagery improves mood soft finish")
        self.assertEqual(group["columns"][1], "Brand imagery improves mood deep clean")

    def test_transpose_result_swaps_rows_and_columns(self) -> None:
        source = {
            "mode": "mr",
            "row_label": "MR options",
            "col_label": "Banner",
            "tables": [
                {
                    "kind": "counts",
                    "columns": ["Male", "Female"],
                    "rows": [
                        {"__label__": "Option A", "Male": 3, "Female": 4},
                        {"__label__": "Option B", "Male": 5, "Female": 6},
                    ],
                }
            ],
        }
        result = _transpose_result(source)
        self.assertEqual(result["row_label"], "Banner")
        self.assertEqual(result["col_label"], "MR options")
        self.assertEqual(result["tables"][0]["columns"], ["Option A", "Option B"])
        self.assertEqual(result["tables"][0]["rows"][0]["__label__"], "Male")
        self.assertEqual(result["tables"][0]["rows"][0]["Option A"], 3)

    def test_transpose_result_marks_base_column_as_numeric(self) -> None:
        source = {
            "mode": "mr",
            "row_label": "MR options",
            "col_label": "Banner",
            "tables": [
                {
                    "kind": "rowpct",
                    "columns": ["Male", "Female"],
                    "rows": [
                        {"__label__": "Base", "Male": 12, "Female": 8},
                        {"__label__": "Option A", "Male": 25.0, "Female": 75.0},
                    ],
                }
            ],
        }
        result = _transpose_result(source)
        self.assertEqual(result["tables"][0]["special_count_columns"], ["Base"])
        self.assertEqual(result["tables"][0]["rows"][0]["Base"], 12)

    def test_mr_preview_uses_option_text_and_group_stem(self) -> None:
        text_df = pd.DataFrame(
            {
                "Q4: Work industry | Advertising Agency": ["Checked", "", "Checked"],
                "Q4: Work industry | Grocery stores": ["", "Checked", ""],
                "Banner": ["Male", "Female", "Male"],
            }
        )
        value_df = pd.DataFrame(
            {
                "Q4: Work industry | Advertising Agency": [1, 0, 1],
                "Q4: Work industry | Grocery stores": [0, 1, 0],
                "Banner": [1, 2, 1],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        mr_detection = detect_mr_groups(value_df)
        result = tabulate_mr(
            value_df,
            bundle=bundle,
            mr_cols=[
                "Q4: Work industry | Advertising Agency",
                "Q4: Work industry | Grocery stores",
            ],
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            mr_detection=mr_detection,
        )
        self.assertEqual(result["row_label"], "Q4 : Work industry")
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["Advertising Agency", "Grocery stores"])

    def test_mr_show_row_codes_still_shows_response_only(self) -> None:
        text_df = pd.DataFrame(
            {
                "Q4: Work industry | Advertising Agency": ["Checked", "", "Checked"],
                "Q4: Work industry | Grocery stores": ["", "Checked", ""],
                "Banner": ["Male", "Female", "Male"],
            }
        )
        value_df = pd.DataFrame(
            {
                "Q4: Work industry | Advertising Agency": [1, 0, 1],
                "Q4: Work industry | Grocery stores": [0, 1, 0],
                "Banner": [1, 2, 1],
            }
        )
        bundle = _build_question_mappings(text_df, value_df)
        mr_detection = detect_mr_groups(value_df)
        result = tabulate_mr(
            value_df,
            bundle=bundle,
            mr_cols=[
                "Q4: Work industry | Advertising Agency",
                "Q4: Work industry | Grocery stores",
            ],
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            show_row_codes=True,
            mr_detection=mr_detection,
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["Advertising Agency", "Grocery stores"])

    def test_codebook_mapping_preserves_explicit_order(self) -> None:
        codebook_df = pd.DataFrame(
            {
                "question": ["Q1", "Q1", "Q1", "Banner", "Banner"],
                "code": [2, 3, 1, 10, 20],
                "label": ["Agree", "Neutral", "Disagree", "Male", "Female"],
                "order": [2, 3, 1, 2, 1],
            }
        )
        bundle = _build_question_mappings_from_codebook(codebook_df)
        q1_labels = [category.label for category in bundle.questions["Q1"].categories]
        banner_labels = [category.label for category in bundle.questions["Banner"].categories]
        self.assertEqual(q1_labels, ["Disagree", "Agree", "Neutral"])
        self.assertEqual(banner_labels, ["Female", "Male"])

    def test_codebook_mapping_rejects_duplicate_orders(self) -> None:
        codebook_df = pd.DataFrame(
            {
                "question": ["Q1", "Q1"],
                "code": [1, 2],
                "label": ["Disagree", "Agree"],
                "order": [1, 1],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate order"):
            _build_question_mappings_from_codebook(codebook_df)

    def test_load_mapping_reads_codebook_layout(self) -> None:
        base = Path("tests") / "_tmp_codebook_mapping"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        try:
            text_path = base / "dataset.xlsx"
            value_path = base / "Value.xlsx"
            self.value_df.to_excel(text_path, index=False)
            pd.DataFrame(
                {
                    "Question": ["Q1", "Q1", "Q1", "Banner", "Banner"],
                    "Code": [2, 3, 1, 10, 20],
                    "Label": ["Agree", "Neutral", "Disagree", "Male", "Female"],
                    "Order": [2, 3, 1, 2, 1],
                }
            ).to_excel(value_path, index=False)

            bundle = load_mapping(str(text_path), str(value_path))

            self.assertEqual(bundle.source_kind, "codebook")
            self.assertEqual(
                [category.label for category in bundle.questions["Q1"].categories],
                ["Disagree", "Agree", "Neutral"],
            )
        finally:
            if base.exists():
                shutil.rmtree(base)

    def test_tabulate_sr_uses_codebook_order_not_numeric_sort(self) -> None:
        codebook_df = pd.DataFrame(
            {
                "question": ["Q1", "Q1", "Q1", "Banner", "Banner"],
                "code": [2, 3, 1, 10, 20],
                "label": ["Agree", "Neutral", "Disagree", "Male", "Female"],
                "order": [2, 3, 1, 2, 1],
            }
        )
        bundle = _build_question_mappings_from_codebook(codebook_df)
        result = tabulate_sr(
            self.value_df,
            bundle=bundle,
            row_var="Q1",
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=False,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            row_sort_mode="code_desc",
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        columns = result["tables"][0]["columns"]
        self.assertEqual(labels, ["Disagree", "Agree", "Neutral"])
        self.assertEqual(columns, ["Female", "Male"])

    def test_tabulate_mr_uses_codebook_option_order(self) -> None:
        df = pd.DataFrame(
            {
                "Q4: Work industry | Advertising Agency": [1, 0, 1],
                "Q4: Work industry | Grocery stores": [0, 1, 0],
                "Q4: Work industry | Market Research": [0, 0, 0],
                "Banner": [1, 2, 1],
            }
        )
        codebook_df = pd.DataFrame(
            {
                "question": [
                    "Q4 : Work industry",
                    "Q4 : Work industry",
                    "Q4 : Work industry",
                    "Banner",
                    "Banner",
                ],
                "code": ["grocery", "advertising", "research", 2, 1],
                "label": ["Grocery stores", "Advertising Agency", "Market Research", "Female", "Male"],
                "order": [1, 2, 3, 1, 2],
            }
        )
        bundle = _build_question_mappings_from_codebook(codebook_df)
        mr_detection = detect_mr_groups(df)
        result = tabulate_mr(
            df,
            bundle=bundle,
            mr_cols=[
                "Q4: Work industry | Advertising Agency",
                "Q4: Work industry | Grocery stores",
                "Q4: Work industry | Market Research",
            ],
            col_var="Banner",
            col_var2=None,
            include_totals=False,
            include_base=True,
            out_counts=True,
            out_rowpct=False,
            out_colpct=False,
            mr_detection=mr_detection,
        )
        labels = [row["__label__"] for row in result["tables"][0]["rows"]]
        self.assertEqual(labels, ["Base", "Grocery stores", "Advertising Agency", "Market Research"])
        self.assertEqual(result["tables"][0]["rows"][3]["Female"], 0)


if __name__ == "__main__":
    unittest.main()
