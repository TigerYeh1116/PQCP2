import gzip
import json
import zlib

import pytest

from experiments.recover_coarse_ledgers import recover_records


def test_missing_footer_preserves_complete_records_without_claiming_crc(tmp_path):
    data=b'{"anchor":0}\n{"anchor":1}\n'
    source=tmp_path/'original.gz';source.write_bytes(gzip.compress(data)[:-8])
    original=source.read_bytes();destination=tmp_path/'recovered.gz'
    result=recover_records(source,destination)
    assert gzip.decompress(destination.read_bytes())==data
    assert not result['original_gzip_footer_verified'] and result['complete_json_records']==2
    assert source.read_bytes()==original


def test_unterminated_record_is_not_credited(tmp_path):
    data=b'{"anchor":0}\n{"anchor":'
    source=tmp_path/'original.gz';source.write_bytes(gzip.compress(data)[:-8])
    destination=tmp_path/'recovered.gz';result=recover_records(source,destination)
    assert gzip.decompress(destination.read_bytes())==b'{"anchor":0}\n'
    assert result['complete_json_records']==1 and result['discarded_unterminated_bytes']==10


def test_corrupt_crc_and_complete_invalid_json_fail(tmp_path):
    source=tmp_path/'original.gz';packed=bytearray(gzip.compress(b'{"anchor":0}\n'))
    packed[-8]^=1;source.write_bytes(packed)
    with pytest.raises(zlib.error):recover_records(source,tmp_path/'bad_crc.gz')
    source.write_bytes(gzip.compress(b'not json\n'))
    with pytest.raises(json.JSONDecodeError):recover_records(source,tmp_path/'bad_json.gz')
