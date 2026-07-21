from pathlib import Path

from passport_service.core.mrz import read_passport_file_mrz
from tests import TEST_DATA_DIR

TEST_DIR = Path(__file__).parent.joinpath("data")


def test_read_mrz() -> None:
    # Given
    test_image_path = TEST_DATA_DIR.joinpath("passports", "passport.png")
    # When
    mrz = read_passport_file_mrz(str(test_image_path))
    # Then
    assert mrz.names == "JANE"
    assert mrz.surname == "SMITH"
    assert mrz.country == "EOL"
