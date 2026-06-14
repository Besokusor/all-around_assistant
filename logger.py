"""
Trace 日志系统 — 持久化各流程执行轨迹

双通道日志：
  1. assistant.log        — 摘要日志（流程进度、关键指标）
  2. assistant.trace.log  — 原始数据日志（每环节的输入/输出原始内容）

特性：
  - 按日期轮转的日志文件，保留 30 天
  - Trace ID 贯穿整个请求链路（ContextVar 线程/协程隔离）
  - 五层架构标签：memory / think / exec / observe / output / fallback
  - 工具调用耗时、重试次数、降级级别等关键指标
  - 控制台同步输出 + 文件持久化

输出文件（位于 ./log/ 目录）：
  - assistant.log        全量摘要日志
  - assistant.trace.log  原始数据 trace（每个环节的输入/输出）
  - assistant.error.log  仅 ERROR 及以上
"""
import json
import logging
import logging.handlers
import os
import sys
import time
from contextvars import ContextVar
from typing import Optional, Any

# ==================== Trace ID 上下文 ====================
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

# ==================== 自定义 Formatter ====================

LOG_LAYER_LABELS = {
    "memory":   "🧠 memory  ",
    "think":    "💡 think   ",
    "exec":     "⚡ exec    ",
    "observe":  "👁️ observe ",
    "output":   "💬 output  ",
    "fallback": "🆘 fallback",
    "agent":    "🤖 agent   ",
    "server":   "🌐 server  ",
}


class SummaryFormatter(logging.Formatter):
    """摘要日志格式：时间 | 级别 | [trace:xxx] | [层标签] | 消息"""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = _trace_id.get()
        trace_part = f"[{trace_id}]" if trace_id else "[---------]"

        layer = getattr(record, "layer", "agent")
        layer_label = LOG_LAYER_LABELS.get(layer, f"[{layer:8}]")

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        ms = int((record.created - int(record.created)) * 1000)
        ts_ms = f"{ts}.{ms:03d}"

        return (
            f"{ts_ms} | {record.levelname:<5} | {trace_part} "
            f"| {layer_label} | {record.getMessage()}"
        )


class TraceFormatter(logging.Formatter):
    """原始数据 Trace 格式：时间 | [trace:xxx] | [层标签] | IN/OUT | JSON 数据"""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = _trace_id.get()
        trace_part = f"[{trace_id}]" if trace_id else "[---------]"

        layer = getattr(record, "layer", "agent")
        layer_label = LOG_LAYER_LABELS.get(layer, f"[{layer:8}]")

        direction = getattr(record, "direction", "OUT")
        direction_label = "IN " if direction == "IN" else "OUT"

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        ms = int((record.created - int(record.created)) * 1000)
        ts_ms = f"{ts}.{ms:03d}"

        return (
            f"{ts_ms} | {trace_part} | {layer_label} | {direction_label} | "
            f"{record.getMessage()}"
        )


# ==================== 日志管理器 ====================

class TraceLogger:
    """Trace 日志管理器 — 单例模式，双通道输出"""

    _instance: Optional["TraceLogger"] = None

    def __init__(self, log_dir: str = "./log", level: str = "INFO",
                 console: bool = True):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # ===== 摘要 Logger =====
        self.logger = logging.getLogger("assistant")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False

        summary_fmt = SummaryFormatter()

        # 文件：全量摘要
        all_path = os.path.join(self.log_dir, "assistant.log")
        file_h = logging.handlers.TimedRotatingFileHandler(
            all_path, when="midnight", interval=1,
            backupCount=30, encoding="utf-8",
        )
        file_h.setLevel(logging.DEBUG)
        file_h.setFormatter(summary_fmt)
        self.logger.addHandler(file_h)

        # 文件：仅 ERROR
        err_path = os.path.join(self.log_dir, "assistant.error.log")
        err_h = logging.handlers.TimedRotatingFileHandler(
            err_path, when="midnight", interval=1,
            backupCount=30, encoding="utf-8",
        )
        err_h.setLevel(logging.ERROR)
        err_h.setFormatter(summary_fmt)
        self.logger.addHandler(err_h)

        # 控制台
        if console:
            console_h = logging.StreamHandler(sys.stdout)
            console_h.setLevel(logging.DEBUG)
            console_h.setFormatter(summary_fmt)
            self.logger.addHandler(console_h)

        # ===== 原始数据 Trace Logger（独立 channel） =====
        self.trace_logger = logging.getLogger("assistant.trace")
        self.trace_logger.setLevel(logging.DEBUG)
        self.trace_logger.propagate = False

        trace_fmt = TraceFormatter()

        trace_path = os.path.join(self.log_dir, "assistant.trace.log")
        trace_h = logging.handlers.TimedRotatingFileHandler(
            trace_path, when="midnight", interval=1,
            backupCount=30, encoding="utf-8",
        )
        trace_h.setLevel(logging.DEBUG)
        trace_h.setFormatter(trace_fmt)
        self.trace_logger.addHandler(trace_h)

        # 存储引用
        self._log_dir = self.log_dir
        self._summary_handler = file_h
        self._trace_handler = trace_h
        self._err_handler = err_h

    @classmethod
    def get_instance(cls, log_dir: str = "./log", level: str = "INFO") -> "TraceLogger":
        if cls._instance is None:
            cls._instance = cls(log_dir=log_dir, level=level)
        return cls._instance

    # ==================== 摘要日志方法 ====================

    def _log(self, level: int, layer: str, msg: str, **kwargs):
        extra = {"layer": layer}
        self.logger.log(level, msg, extra=extra, **kwargs)

    def memory(self, msg: str):
        self._log(logging.INFO, "memory", msg)

    def think(self, msg: str):
        self._log(logging.INFO, "think", msg)

    def execute(self, msg: str):
        self._log(logging.INFO, "exec", msg)

    def observe(self, msg: str):
        self._log(logging.INFO, "observe", msg)

    def output(self, msg: str):
        self._log(logging.INFO, "output", msg)

    def fallback(self, msg: str):
        self._log(logging.INFO, "fallback", msg)

    def info(self, msg: str, layer: str = "agent"):
        self._log(logging.INFO, layer, msg)

    def warning(self, msg: str, layer: str = "agent"):
        self._log(logging.WARNING, layer, msg)

    def error(self, msg: str, layer: str = "agent"):
        self._log(logging.ERROR, layer, msg)

    def debug(self, msg: str, layer: str = "agent"):
        self._log(logging.DEBUG, layer, msg)

    def tool_call(self, tool_name: str, duration_ms: float, success: bool,
                  result_preview: str = ""):
        status = "✅" if success else "❌"
        preview = result_preview[:100] if result_preview else ""
        self._log(
            logging.INFO, "exec",
            f"tool.{tool_name} → {status} ({duration_ms}ms) {preview}"
        )

    # ==================== 原始数据 Trace 方法 ====================

    def _trace_raw_log(self, layer: str, direction: str, data: Any):
        """
        写入原始数据到 assistant.trace.log

        Args:
            layer: 层标签 (memory/think/exec/observe/output/fallback/agent)
            direction: "IN" 或 "OUT"
            data: 原始数据（dict/list/str — 自动 JSON 序列化）
        """
        extra = {"layer": layer, "direction": direction}

        if isinstance(data, (dict, list)):
            text = json.dumps(data, ensure_ascii=False, default=str)
        else:
            text = str(data)

        self.trace_logger.log(logging.INFO, text, extra=extra)

    def trace_input(self, layer: str, data: Any):
        """记录环节输入（原始数据）"""
        self._trace_raw_log(layer, "IN", data)

    def trace_output(self, layer: str, data: Any):
        """记录环节输出（原始数据）"""
        self._trace_raw_log(layer, "OUT", data)

    def trace_io(self, layer: str, input_data: Any = None, output_data: Any = None):
        """同时记录输入和输出"""
        if input_data is not None:
            self._trace_raw_log(layer, "IN", input_data)
        if output_data is not None:
            self._trace_raw_log(layer, "OUT", output_data)

    # ==================== 请求级便捷方法 ====================

    def trace_request_start(self, user_input: str, user_id: str,
                            image_source: str = None):
        """记录请求开始（摘要 + 原始输入）"""
        input_preview = user_input[:60].replace("\n", " ")
        self._log(
            logging.INFO, "agent",
            f"══════════ 请求开始 │ user={user_id} "
            f"image={'有' if image_source else '无'} │ {input_preview}"
        )
        self._trace_raw_log("agent", "IN", {
            "event": "request_start",
            "user_input": user_input,
            "user_id": user_id,
            "image_source": image_source,
        })

    def trace_request_end(self, total_ms: float, steps: int, tools_called: int,
                          response_len: int, final_response: str = ""):
        """记录请求结束（摘要 + 原始输出）"""
        self._log(
            logging.INFO, "agent",
            f"══════════ 请求完成 │ 总耗时={total_ms:.0f}ms 步骤={steps} "
            f"工具调用={tools_called} 回复长度={response_len}"
        )
        self._trace_raw_log("agent", "OUT", {
            "event": "request_end",
            "total_ms": round(total_ms, 1),
            "steps": steps,
            "tools_called": tools_called,
            "response_length": response_len,
            "final_response": final_response,
        })


# ==================== 便捷函数 ====================

def set_trace_id(trace_id: str):
    _trace_id.set(trace_id)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def init_logger(log_dir: str = "./log", level: str = "INFO") -> TraceLogger:
    """初始化日志系统（幂等，应在应用启动时调用一次）"""
    return TraceLogger.get_instance(log_dir=log_dir, level=level)


def get_logger() -> Optional[TraceLogger]:
    """获取当前 TraceLogger 实例（未初始化则返回 None）"""
    return TraceLogger._instance
