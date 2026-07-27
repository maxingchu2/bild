"""Plugin 层：打包分发（用户安装，配置级生命周期）。

一个 Plugin 把 tools / skills / hooks / goals 打包成一个可安装单元。
内置 "ship-inspection" 插件：船检业务全家桶（当前默认安装）。
"""

from typing import Any, Callable, Dict, List


class Plugin:
    def __init__(self, name: str, version: str, description: str,
                 setup: Callable[[], None]):
        self.name = name
        self.version = version
        self.description = description
        self.setup = setup
        self.installed = False

    def install(self) -> None:
        if not self.installed:
            self.setup()
            self.installed = True


class PluginManager:
    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        self._plugins[plugin.name] = plugin

    def install(self, name: str) -> None:
        self._plugins[name].install()

    def install_all(self) -> None:
        for p in self._plugins.values():
            p.install()

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [{"name": p.name, "version": p.version,
                 "description": p.description, "installed": p.installed}
                for p in self._plugins.values()]


plugin_manager = PluginManager()


def _setup_ship_inspection() -> None:
    # skills / hooks 在各自模块 import 时已注册；这里保留扩展点
    from . import hooks, skills  # noqa: F401


plugin_manager.register(Plugin(
    "ship-inspection", "1.0.0",
    "船检业务插件：15 个业务动作 + 流程 Skill + 审计 Hook",
    _setup_ship_inspection,
))
