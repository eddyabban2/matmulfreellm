from mmfreelm import __version__
from mmfreelm.models import HGRNBitConfig


def test_package_version_is_defined():
    assert __version__


def test_hgrn_bit_config_accepts_custom_dimensions():
    config = HGRNBitConfig(hidden_size=64, num_hidden_layers=1)

    assert config.hidden_size == 64
    assert config.num_hidden_layers == 1
