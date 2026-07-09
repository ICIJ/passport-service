import os

from icij_worker.task_manager.amqp import AMQPTaskManagerConfig

from passport_service.config import HttpServiceConfig


def test_http_config_from_env(reset_env) -> None:  # noqa: ANN001, ARG001
    # Given
    env = {
        "PASSPORT_HTTP_TASK_MANAGER__BACKEND": "amqp",
        "PASSPORT_HTTP_TASK_MANAGER__STORAGE__MAX_CONNECTIONS": "28",
        "PASSPORT_HTTP_TASK_MANAGER__APP_PATH": "some_path",
    }
    os.environ.update(env)
    # When
    config = HttpServiceConfig()
    # Then
    assert isinstance(config.task_manager, AMQPTaskManagerConfig)
