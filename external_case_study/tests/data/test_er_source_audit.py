from pathlib import Path

from src.data.er_source_audit import parse_clinical_xml, resolve_primary_xml


def test_xml_parser_separates_primary_from_follow_up(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
    <brca:tcga_bcr xmlns:brca="urn:brca" xmlns:s="urn:shared">
      <brca:patient>
        <s:bcr_patient_barcode>TCGA-XX-0001</s:bcr_patient_barcode>
        <s:bcr_patient_uuid>AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE</s:bcr_patient_uuid>
        <brca:breast_carcinoma_estrogen_receptor_status cde="2957359"
          preferred_name="er_status_by_ihc">Positive</brca:breast_carcinoma_estrogen_receptor_status>
        <brca:follow_ups><brca:follow_up>
          <brca:breast_carcinoma_estrogen_receptor_status>Negative</brca:breast_carcinoma_estrogen_receptor_status>
        </brca:follow_up></brca:follow_ups>
      </brca:patient>
    </brca:tcga_bcr>"""
    path = tmp_path / "clinical.xml"
    path.write_text(xml, encoding="utf-8")

    observations = parse_clinical_xml(
        path,
        file_id="file-1",
        file_name=path.name,
        expected_case_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )

    assert [(item.context, item.label) for item in observations] == [
        ("primary_patient", 1),
        ("follow_up", 0),
    ]
    resolved = resolve_primary_xml(observations)
    assert resolved.loc[0, "xml_primary_label"] == 1
    assert resolved.loc[0, "xml_nested_er_record_count"] == 1


def test_xml_primary_unknown_is_not_imputed(tmp_path: Path) -> None:
    path = tmp_path / "clinical.xml"
    path.write_text(
        """<root><patient><bcr_patient_barcode>TCGA-XX-0002</bcr_patient_barcode>
        <bcr_patient_uuid>aaaaaaaa-bbbb-cccc-dddd-ffffffffffff</bcr_patient_uuid>
        <breast_carcinoma_estrogen_receptor_status>[Not Available]</breast_carcinoma_estrogen_receptor_status>
        </patient></root>""",
        encoding="utf-8",
    )
    resolved = resolve_primary_xml(
        parse_clinical_xml(path, file_id="f", file_name=path.name)
    )
    assert resolved.loc[0, "xml_primary_reason"] == "xml_primary_unknown"
    assert resolved.loc[0, "xml_primary_label"] is None
