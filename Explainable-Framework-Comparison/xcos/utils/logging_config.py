import logging
import sys

# StreamHandler 控制台输出日志 终端日志
stream_handler = logging.StreamHandler(sys.stdout)
#创建日志格式对象
format_ = ('[%(asctime)s] {%(filename)s:%(lineno)d} '
           '%(levelname)s - %(message)s')

try:
    # use colored logs if installed
    # 如果导入了这个库
    import coloredlogs
    # Colored格式
    formatter = coloredlogs.ColoredFormatter(fmt=format_)
    # StreamHandler对象自定义日志格式
    stream_handler.setFormatter(formatter)
except Exception:
    pass

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format=format_,
    datefmt='%m-%d %H:%M:%S',
    handlers=[stream_handler]
)

# 创建日志的实例
logger = logging.getLogger(__name__)
