import os

from icij_worker.task_manager.amqp import AMQPTaskManagerConfig

from passport_service.config import HttpServiceConfig


def test_http_config_from_env(reset_env) -> None:  # noqa: ANN001, ARG001
    # Given
    env = {
        "PASSPORT_HTTP_TASK_MANAGER__BACKEND": "amqp",
        "PASSPORT_HTTP_TASK_MANAGER__STORAGE__MAX_CONNECTIONS": "28",
    }
    os.environ.update(env)
    # When
    config = HttpServiceConfig.from_env()
    # Then

    assert isinstance(config.task_manager, AMQPTaskManagerConfig)
