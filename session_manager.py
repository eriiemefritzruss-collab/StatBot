import time
from collections import OrderedDict

from statbot import StatBot


class SessionManager:
    def __init__(self, config_path="config.yaml", max_sessions=8):
        self.config_path = config_path
        self.max_sessions = max_sessions
        self._sessions = OrderedDict()

    def get(self, session_id):
        session_key = session_id or f"local-{int(time.time() * 1000)}"
        if session_key in self._sessions:
            bot = self._sessions.pop(session_key)
            self._sessions[session_key] = bot
            return bot

        bot = StatBot(config_path=self.config_path)
        self._sessions[session_key] = bot
        self._evict_if_needed()
        return bot

    def _evict_if_needed(self):
        while len(self._sessions) > self.max_sessions:
            _, bot = self._sessions.popitem(last=False)
            try:
                bot.conv.kernel.shutdown()
            except Exception:
                pass
