import logging
import time
import backoff  # for exponential backoff
import openai
import re
import asyncio
from typing import List, Dict, Any
# --- 导入具体的错误类型以提高代码可读性和健壮性 ---
from openai import OpenAI, BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError
import os
import threading
import random
from dotenv import load_dotenv
from httpx import Timeout
# --- 新增导入：tqdm 用于显示进度条 ---
from tqdm.auto import tqdm

# --- 日志配置 ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- 一个函数，用于判断哪些错误是“永久性”的，不应重试 ---
def _is_permanent_failure(e: Exception) -> bool:
    """
    判断一个 API 错误是否为永久性错误，不应重试。
    这包括客户端错误，如错误的请求 (400)、认证失败 (401)、权限不足 (403)
    和资源未找到 (404)，因为这些问题无法通过重试解决。
    """
    return isinstance(e, (
        openai.BadRequestError,
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
    ))

def do_nothing(para):
    pass

# --- Backoff 装饰器，用于在出现可重试的API错误时自动重试 ---
enhanced_backoff = backoff.on_exception(
    backoff.expo,
    # 捕获可能瞬态的错误（包括所有HTTP状态码错误、连接和超时错误）。
    (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError),
    max_tries=1,
    max_time=180,
    factor=2,
    jitter=backoff.full_jitter,
    # 如果错误是永久性的，则立即停止重试。
    giveup=_is_permanent_failure,
    on_backoff=lambda details: logger.warning(
        f"API call failed with retryable error. "
        f"Backing off {details['wait']:.1f}s after {details['tries']} tries. "
        f"Function: {details['target'].__name__}. Error: {details['exception']}"
    ),
    on_giveup=do_nothing
)


# --- 一个与tqdm绑定的backoff装饰器，用于在批量处理时提供更好的反馈 ---
def create_backoff_decorator_with_tqdm(pbar: tqdm):
    """
    创建一个特殊的 backoff 装饰器，其 on_giveup 回调会更新 tqdm 进度条。
    :param pbar: 一个 tqdm 实例。
    :return: 一个配置好的 backoff 装饰器。
    """
    state = {'permanent_failures': 0}

    def on_giveup_callback(details: Dict[str, Any]):
        """当重试被永久性放弃时调用此函数。"""
        state['permanent_failures'] += 1
        pbar.set_postfix(permanent_failures=state['permanent_failures'], refresh=True)
        logger.error(
            f"API call failed permanently and gave up after {details['tries']} tries. "
            f"Function: {details['target'].__name__}. Error: {details['exception']}"
        )

    def on_backoff_callback(details: Dict[str, Any]):
        """当发生可重试错误并进行退避时调用此函数。"""
        logger.warning(
            f"API call failed with retryable error. "
            f"Backing off {details['wait']:.1f}s after {details['tries']} tries. "
            f"Function: {details['target'].__name__}. Error: {details['exception']}"
        )

    return backoff.on_exception(
        backoff.expo,
        (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError),
        max_tries=1,
        max_time=180,
        factor=2,
        jitter=backoff.full_jitter,
        giveup=_is_permanent_failure,
        on_backoff=on_backoff_callback,
        on_giveup=on_giveup_callback
    )


class ModelLoader:
    """
    一个有状态的模型加载器，内置了线程安全的后台热更新功能。
    """
    def __init__(self, dot_env_path, interval=60):
        """
        初始化模型加载器。
        :param dot_env_path: .env文件的路径。
        :param interval: 更新模型列表的时间间隔（秒）。
        """
        self.dot_env_path = dot_env_path
        self.MODEL_NAME = ""
        self._models: List['OpenAIModel'] = []
        self._lock = threading.Lock()
        self.interval = interval
        self._stop_event = threading.Event()

        # 1. 创建并管理一个专用的 asyncio 事件循环，运行在后台线程中
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True, name="AsyncLoopThread")
        self.loop_thread.start()
        logger.warning("后台 asyncio 事件循环线程已启动。")
        
        # 2. 首次加载模型
        logger.warning("首次加载和测试模型...")
        self._update_models()

        # 3. 启动后台热更新线程
        self._update_thread = threading.Thread(target=self._periodic_update, daemon=True, name="ModelUpdateThread")
        self._update_thread.start()
        logger.warning(f"模型热加载器已启动，每 {self.interval} 秒更新一次。")

    def _run_loop(self):
        """后台线程的目标函数，用于运行专用的事件循环。"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def _update_models(self):
        """从环境变量加载并测试模型，仅保留测试成功的模型。"""
        try:
            new_models, client_args = load_openai_models_from_ENV(
                self.dot_env_path
            )
            
            if new_models:
                with self._lock:
                    self._models = new_models
                    self.MODEL_NAME = [cur_client_arg['model_name']  if 'model_name' in cur_client_arg and cur_client_arg['model_name'] else "" for cur_client_arg in client_args][0]
                    self.REQUEST_TIMEOUT = [cur_client_arg['timeout'] if 'timeout' in cur_client_arg and cur_client_arg['timeout'] else 180 for cur_client_arg in client_args][0]
                    self.API_KEYS = [cur_client_arg['api_key'] if 'api_key' in cur_client_arg else None for cur_client_arg in client_args]
                    self.BASE_URLS = [cur_client_arg['base_url'] if 'base_url' in cur_client_arg else None for cur_client_arg in client_args]

                    for cur_model, cur_api_key, cur_base_url in zip(self._models, self.API_KEYS, self.BASE_URLS):
                        cur_model.request_timeout = self.REQUEST_TIMEOUT
                        cur_model.base_url = cur_base_url
                        cur_model.api_key = cur_api_key

                logger.warning(f"模型列表已成功更新。当前可用模型数量: {len(self._models)}, 所有 API 请求将使用 {self.REQUEST_TIMEOUT} 秒的超时时间。")
            else:
                logger.warning("本次热更新未能找到任何可用的模型。为保证服务连续性，将保留旧的模型列表。")
        except Exception as e:
            logger.error(f"热加载模型过程中发生意外错误: {e}", exc_info=True)

    def _periodic_update(self):
        """后台线程运行的函数，定期更新模型。"""
        while not self._stop_event.wait(self.interval):
            self._update_models()

    def get_random_model(self) -> 'OpenAIModel':
        """线程安全地获取一个随机模型实例。"""
        with self._lock:
            if not self._models:
                raise ValueError("模型列表为空，无法获取模型。请检查环境变量和配置。")
            return random.choice(self._models)
    
    def get_random_info(self):
        """线程安全地获取一个随机模型实例及其相关信息。"""
        with self._lock:
            if not self._models:
                raise ValueError("模型列表为空，无法获取模型。请检查环境变量和配置。")
            random_index = random.randint(0, len(self._models) - 1)
            return self._models[random_index], self.API_KEYS[random_index], self.BASE_URLS[random_index], self.MODEL_NAME

    def stop(self):
        """优雅地停止所有后台线程。"""
        if self._stop_event.is_set():
            return
        logger.warning("正在停止模型热加载器和事件循环...")
        
        self._stop_event.set()
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=2)
        
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2)
            
        logger.warning("模型热加载器和事件循环已停止。")


def load_openai_models_from_ENV(dot_env_path: str):
    """从.env文件加载、测试并返回注入了依赖的模型实例列表。"""
    if not os.path.exists(dot_env_path):
        logger.error(f"严重错误：指定的.env文件路径不存在: '{dot_env_path}'。")
        return [], None

    load_dotenv(dot_env_path, override=True)
    
    api_keys_str = os.getenv("API_KEYS")
    model_name = os.getenv("MODEL_NAME")
    try:
        request_timeout = int(os.getenv("REQUEST_TIMEOUT", 360))
    except (ValueError, TypeError):
        logger.warning("REQUEST_TIMEOUT 环境变量格式错误，将使用默认值 360 秒。")
        request_timeout = 360
    
    if not api_keys_str:
        logger.error(f"错误：未能从 '{dot_env_path}' 文件中加载到 'API_KEYS'。")
        return [], None

    base_urls_str = os.getenv("BASE_URLS", "")

    base_urls = [url.strip() for url in re.split(r',|\n', base_urls_str) if url.strip()]
    api_keys = [key.strip() for key in re.split(r',|\n', api_keys_str) if key.strip()]

    if base_urls and len(base_urls) != len(api_keys):
        logger.error(f"配置错误：BASE_URLS ({len(base_urls)}个) 和 API_KEYS ({len(api_keys)}个) 数量不匹配。")
        return [], None

    successful_models = []
    args_list = []
    endpoints = zip(base_urls, api_keys) if base_urls else [(None, key) for key in api_keys]
    
    for base_url, api_key in list(endpoints):
        display_url = base_url if base_url else "Official OpenAI"
        try:
            client_args = {'api_key': api_key.strip()}
            if base_url:
                client_args['base_url'] = base_url.strip()
            
            client = OpenAI(**client_args)
            client.models.list()  # Test call

            args_list.append({
                'api_key': api_key.strip(), 
                'timeout': request_timeout,
                'model_name': model_name,
                'base_url': base_url.strip() if base_url else None
            })
            
            openai_model = OpenAIModel(client=client)
            successful_models.append(openai_model)
            # tqdm.write(f"✓ API Key for {display_url} is valid.")
        except Exception as e:
            tqdm.write(f"✗ API Key for {display_url} failed test: {e}")
            continue
    
    if not successful_models:
        logger.warning("所有API Key均测试失败或未提供，本次未加载任何模型。")

    return successful_models, args_list


class OpenAIModel:
    """此类使用一个共享的后台事件循环来处理所有异步操作，确保线程安全。"""
    def __init__(self, client: OpenAI):
        self.client = client
        self.request_timeout = None
        self.base_url = None
        self.api_key = None

    def _handle_api_error(self, e: Exception, func_name: str):
        """有条件地记录API错误并重新引发它。"""
        if not _is_permanent_failure(e):
            masked_key = f"...{self.client.api_key[-4:]}" if self.client.api_key else "None"
            logger.warning(
                f"Unexpected API call error in {func_name}: {e}, "
                f"BASE_URL: {self.client.base_url}, API_KEY: {masked_key}"
            )
        raise e

    # @enhanced_backoff
    def _completions(self, model_name, **kwargs):
        """同步的 completions 方法。"""
        try:
            response = self.client.completions.create(model=model_name, **kwargs)
            return response.choices[0].text, response.choices[0].finish_reason
        except Exception as e:
            self._handle_api_error(e, "_completions")

    def completions(self, model_name, **kwargs):
        """[已保留] 调用旧版的 completions API。"""
        return self._completions(model_name, **kwargs)

    # @enhanced_backoff
    def _chat_completions(self, model_name, messages, **kwargs):
        """同步的 chat completions 方法。"""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content.strip(), response.choices[0].finish_reason
        except Exception as e:
            self._handle_api_error(e, "_chat_completions")
    
    def chat_completions(self, model_name, messages, **kwargs):
        return self._chat_completions(model_name, messages, **kwargs)

    async def _dispatch_batch(self, messages_list, model, **kwargs):
        """在异步环境中调度批量请求，每个请求都带重试和tqdm进度条。"""
        pbar = tqdm(total=len(messages_list), desc="Batch Processing", unit="req", dynamic_ncols=True)
        pbar.set_postfix(permanent_failures=0, refresh=True)
        tqdm_enhanced_backoff = create_backoff_decorator_with_tqdm(pbar)

        @tqdm_enhanced_backoff
        async def single_api_call_with_retry(messages):
            """封装单个API调用，包含重试和进度更新。"""
            try:
                # 将同步的SDK调用移到线程池中执行，避免阻塞事件循环
                return await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model, messages=messages, **kwargs
                )
            finally:
                pbar.update(1)

        tasks = [single_api_call_with_retry(msg) for msg in messages_list]
        try:
            return await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            pbar.close()
    
    # @enhanced_backoff
    def embed(self, model_name, input_string, **kwargs):
        try:
            response = self.client.embeddings.create(input=input_string, model=model_name, **kwargs)
            return response.data[0].embedding
        except Exception as e:
            self._handle_api_error(e, "embed")

    # @enhanced_backoff
    def batch_embed(self, model_name, input_string_list, **kwargs):
        try:
            response = self.client.embeddings.create(input=input_string_list, model=model_name, **kwargs)
            return [x.embedding for x in response.data]
        except Exception as e:
            self._handle_api_error(e, "batch_embed")
