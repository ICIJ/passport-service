from icij_common.pydantic_utils import merge_configs
from icij_worker.http_.config import HttpServiceConfig as BaseHttpServiceConfig
from pydantic_settings import SettingsConfigDict


class HttpServiceConfig(BaseHttpServiceConfig):
    model_config = merge_configs(
        BaseHttpServiceConfig.model_config,
        SettingsConfigDict(
            env_prefix="PASSPORT_HTTP_",
            env_nested_delimiter="__",
            nested_model_default_partial_update=True,
            case_sensitive=False,
        ),
    )

    gunicorn_workers: int = 1
