from __future__ import annotations

import argparse
import logging
import sys
import traceback

from fastapi import FastAPI
from gunicorn.app.base import BaseApplication
from icij_common.logging_utils import setup_loggers

import passport_service
from passport_service.config import HttpServiceConfig
from passport_service.http_.service import create_service


class Formatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=35, width=150)


def _start_app_(ns: argparse.Namespace) -> None:
    _start_app(config_path=ns.config_path)


class GunicornApp(BaseApplication):
    def __init__(self, app: FastAPI, config: HttpServiceConfig, **kwargs):
        self.application = app
        self._app_config = config
        super().__init__(**kwargs)

    def load_config(self) -> None:
        self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")
        self.cfg.set("workers", self._app_config.gunicorn_workers)
        bind = f":{self._app_config.port}"
        self.cfg.set("bind", bind)

    def load(self) -> FastAPI:
        return self.application

    @classmethod
    def from_config(cls, config: HttpServiceConfig) -> GunicornApp:
        fast_api = create_service(config)
        return cls(fast_api, config)


def _start_app(config_path: str | None = None) -> None:
    if config_path is not None:
        config = HttpServiceConfig.parse_file(config_path)
    else:
        config = HttpServiceConfig.from_env()
    app = GunicornApp.from_config(config)
    app.run()


def get_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Passport HTTP server start CLI", formatter_class=Formatter
    )
    arg_parser.add_argument("--config-path", type=str)
    arg_parser.set_defaults(func=_start_app_)
    return arg_parser


def main() -> None:
    # Setup loggers temporarily before loggers init using the app configuration

    setup_loggers(["__main__", passport_service.__name__])
    logger = logging.getLogger(__name__)
    try:
        arg_parser = get_arg_parser()
        args = arg_parser.parse_args()

        if hasattr(args, "func"):
            args.func(args)
        else:
            arg_parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt as e:
        logger.error("Application shutdown...")
        raise e
    except Exception as e:
        error_with_trace = "".join(traceback.format_exception(None, e, e.__traceback__))
        logger.error("Error occurred at application startup:\n%s", error_with_trace)
        raise e
