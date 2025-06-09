from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import grpc
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import paramiko
import psutil
import random
import statistics
import sys
import threading
import time
import uuid
import wave

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.abspath('../tts/server/packages/ziax-normalizer/tests'))
import samples
from tts_service_pb2 import AudioFormatOptions, Hints, RawAudio, UtteranceSynthesisRequest
from tts_service_pb2_grpc import TtsServiceStub


# duration for 1 load_level
DURATION = 300  # in s.
LOAD_LEVELS = [1, 5, 10]
EXPAND_COEF = 4
VOICES = ['elena', 'egor', 'egor_good', 'olga', 'olga_good']

HOSTNAME = "address"
PORT = "port"

USERNAME = "user"
KEY_PATH = "ssh_key"
__SSH_PORT = 22


def sort_samples_by_length(phrases):
    short_phrases = []
    medium_phrases = []
    long_phrases = []

    for phrase in phrases:
        length = len(phrase)
        if length < 60:
            short_phrases.append(phrase)
        elif 60 <= length < 120:
            medium_phrases.append(phrase)
        else:
            long_phrases.append(phrase)

    return short_phrases, medium_phrases, long_phrases


def init_test_samples():
    all_samples = [
        samples.address_samples,
        samples.address_new_samples,
        samples.phone_samples,
        samples.personal_samples,
        samples.ordinal_samples,
        samples.decimal_samples,
        samples.time_samples,
        samples.date_samples,
        samples.date_new_samples,
        samples.ordinal_date_samples,
        samples.measure_samples,
        samples.prep_samples,
        samples.latin_samples,
        samples.number_samples,
    ]

    phrases = []
    for sample in all_samples:
        if type(sample) == dict:
            phrases.extend(sample.keys())
            phrases.extend(sample.values())
        else:
            phrases.extend(sample)

    short_phrases, medium_phrases, long_phrases = sort_samples_by_length(phrases)

    # Можно выбрать что-то одно
    return short_phrases + medium_phrases + long_phrases


def expand_and_mix_phrases(phrases, expanding_factor=1.0):
    if expanding_factor < 1.0:
        raise ValueError("expanding_factor must be >= 1.0")

    repeat_factor = math.floor(expanding_factor)
    additional_ratio = expanding_factor - repeat_factor

    expanded_phrases = []
    for phrase in phrases:
        new_phrase = " ".join([phrase] * repeat_factor)
        expanded_phrases.append(new_phrase)

        if random.random() < additional_ratio:
            expanded_phrases.append(new_phrase)
    return expanded_phrases


def send_request(text, endpoint='localhost:5001', voice='elena') -> (int, int, int, bool):
    channel = grpc.insecure_channel(endpoint)
    stub = TtsServiceStub(channel)
    hints = [
        Hints(voice=voice),
    ]

    audio_format = AudioFormatOptions(
        raw_audio=RawAudio(
            audio_encoding=RawAudio.LINEAR16_PCM,
            sample_rate_hertz=8000
        )
    )

    text_length = len(text)
    start_time = time.perf_counter()
    phrase_failed = False
    try:
        with wave.open('output.wav', 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)

            logging.info(f"Starting synthesis for text: '{text[:30]}'... (length: {len(text)} characters)")

            try:
                it = stub.UtteranceSynthesis(
                    UtteranceSynthesisRequest(text=text, hints=hints, output_audio_spec=audio_format),
                    metadata=(('x-client-id', str(uuid.uuid4())),)
                )
                first_response = next(it, None)
                if first_response is None:
                    raise ValueError("No audio data received from the server.")

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                    logging.error(f"Too many requests: {e}")
                    time.sleep(3)
                    return text_length, 0, 0, True

                logging.error(f"Synthesis error: {e}")
                return text_length, 0, 0, True

            except ValueError as e:
                logging.error(f"Unexpected value error: {e}")
                return text_length, 0, 0, True

            writer.writeframes(first_response.audio_chunk.data)
            first_chunk_response_time_duration = time.perf_counter() - start_time

            if first_chunk_response_time_duration > 5:
                logging.error(f"{text[:30]}: First audio chunk received in {first_chunk_response_time_duration:.2f} sec.")
                return text_length, first_chunk_response_time_duration, first_chunk_response_time_duration, True

            logging.info(f"{text[:30]}: First audio chunk received in {first_chunk_response_time_duration:.2f} sec.")

            for response in it:
                writer.writeframes(response.audio_chunk.data)
                if response.audio_chunk.final:
                    break

    except Exception as err:
        phrase_failed = True
        logging.error(f"Unexpected error: {err}")

    full_response_duration = time.perf_counter() - start_time
    logging.info(f"Full request execution time: {full_response_duration:.6f} seconds")
    return text_length, first_chunk_response_time_duration, full_response_duration, phrase_failed


def run_client(thread_id, phrases, result_list, duration):
    end_time = time.time() + duration
    failed_count = 0

    while time.time() < end_time:
        phrase = random.choice(phrases)
        voice = random.choice(VOICES)
        length, response_time, request_duration, phrase_failed = send_request(phrase,
                                                                              endpoint=f'{HOSTNAME}:{PORT}',
                                                                              voice=voice)
        result = {
            'thread_id': thread_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'length': length,
            'duration': request_duration,
            'phrase_failed': phrase_failed,
            'response_time': response_time
        }
        result_list.append(result)
        time.sleep(0.1)

    return failed_count


def __convert_memory_to_mb(mem_usage: str):
    if "MiB" in mem_usage:
        return float(mem_usage.replace("MiB", ""))
    elif "GiB" in mem_usage:
        return float(mem_usage.replace("GiB", "")) * 1024
    elif "KiB" in mem_usage:
        return float(mem_usage.replace("KiB", "")) / 1024
    elif "B" in mem_usage:
        return float(mem_usage.replace("B", "")) / (1024 * 1024)
    return None


def get_remote_docker_stats(ssh, resource_type: str):
    commands = {
        'cpu': "docker stats tts --no-stream --format '{{json .CPUPerc}}'",
        'ram': "docker stats tts --no-stream --format '{{json .MemUsage}}'"
    }

    command = commands.get(resource_type)
    if not command:
        raise ValueError(f"Invalid resource type: {resource_type}")

    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        result = stdout.read().decode().strip()

        if not result:
            raise RuntimeError(f"No data returned for {resource_type} via SSH")

        if resource_type == "cpu":
            return float(result.replace('"', '').replace('%', ''))
        elif resource_type == "ram":
            mem_usage = result.replace('"', '').split(" / ")[0]
            return __convert_memory_to_mb(mem_usage)

    except paramiko.SSHException as e:
        raise RuntimeError(f"SSH error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to get remote {resource_type} stats.")


def get_local_stats(resource_type: str):
    try:
        if resource_type == "cpu":
            usage_per_core = psutil.cpu_percent(interval=0.1, percpu=False)
            num_cpus = psutil.cpu_count(logical=True)
            return usage_per_core * num_cpus
        elif resource_type == "ram":
            mem = psutil.virtual_memory()
            return mem.used / (1024 * 1024)
        else:
            raise ValueError(f"Invalid resource type: {resource_type}")
    except Exception as e:
        logging.error(f"Local stat error for {resource_type}: {e}")
        return None


def monitor_resources(stop_event, cpu_usages, memory_usages, monitoring_timestamps, ssh):
    while not stop_event.is_set():
        cpu_usage = _try_get_stat('cpu', ssh)
        if cpu_usage is not None:
            cpu_usages.append(cpu_usage)

        ram_usage = _try_get_stat('ram', ssh)
        if ram_usage is not None:
            memory_usages.append(ram_usage)

        monitoring_timestamps.append(datetime.now())
        time.sleep(1)


def _try_get_stat(resource_type: str, ssh):
    try:
        return get_remote_docker_stats(ssh, resource_type)
    except Exception as e:
        logging.warning(f"Remote {resource_type.upper()} stat failed, falling back to local: {e}")
        return get_local_stats(resource_type)


def perform_test(requests_num, duration, hostname, port, username, key_path):
    result_list = []
    cpu_usages = []
    memory_usages = []
    monitoring_timestamps = []
    stop_event = threading.Event()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, key_filename=key_path, banner_timeout=200)
    _, stdout, _ = ssh.exec_command('nproc')
    cpus_number = int(stdout.read().decode().strip())

    monitor_thread = threading.Thread(target=monitor_resources, args=(
        stop_event, cpu_usages, memory_usages, monitoring_timestamps, ssh))
    monitor_thread.start()

    init_phrases = init_test_samples()
    phrases = expand_and_mix_phrases(init_phrases, EXPAND_COEF)

    with ThreadPoolExecutor(max_workers=requests_num) as executor:
        future_to_thread = {}
        for i in range(requests_num):
            future = executor.submit(run_client, i, phrases, result_list, duration)
            future_to_thread[future] = i

    stop_event.set()
    monitor_thread.join()
    ssh.close()

    durations = [result['duration'] for result in result_list]
    lengths = [result['length'] for result in result_list]
    timestamps = [result['timestamp'] for result in result_list]
    failed = [result['phrase_failed'] for result in result_list]
    response_time = [result['response_time'] for result in result_list]

    stats = {
        'max_duration': max(durations),
        'min_duration': min(durations),
        'avg_duration': statistics.mean([d for d in durations if d > 0]) if any(d > 0 for d in durations) else 0,
        'avg_length': statistics.mean([l for l in lengths]),
        'nproc': cpus_number,
        'cpu_usages': [usage / cpus_number for usage in cpu_usages],
        'cpu_usages_int': [usage / 100 for usage in cpu_usages],
        'memory_usages': memory_usages,
        'monitoring_timestamps': monitoring_timestamps,
        'lengths': lengths,
        'durations': durations,
        'timestamps': timestamps,
        'failed': failed,
        'avg_response_time': statistics.mean([rt for rt in response_time if rt > 0]) if any(rt > 0 for rt in response_time) else 0,
    }

    return stats


def show_statistics(results):
    print("=== Statistics ===")
    print("Threads | % failed | Min. len failed | Max len. success")

    recommended_max_threads = 1
    recommended_max_length = 0

    for load_level, stats in results:
        total_requests = len(stats['failed'])
        failed_count = sum(stats['failed'])

        fail_rate = (failed_count / total_requests) * 100 if total_requests > 0 else 0

        failed_lengths = [l for l, f in zip(stats['lengths'], stats['failed']) if f]
        success_lengths = [l for l, f in zip(stats['lengths'], stats['failed']) if not f]

        min_failed_length = min(failed_lengths) if failed_lengths else 0
        max_success_length = max(success_lengths) if success_lengths else 0

        print(f"{load_level:6} | {fail_rate:8.2f}% | {min_failed_length:16} | {max_success_length:14}")

        # only if fail rate for this thread count <= 10
        if fail_rate <= 10 and load_level >= recommended_max_threads:
            recommended_max_threads = load_level
            recommended_max_length = max_success_length

    print("\nRecommended limitation:")
    print(f"- {recommended_max_threads} threads")
    print(f"- {recommended_max_length} symbols")


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def __main():
    """
    Если результаты недостоверны, элементы на графике будут выделены красным.
    """
    results = []

    for load_level in LOAD_LEVELS:
        logging.info(f"Testing with {load_level} concurrent requests...")
        stats = perform_test(load_level, DURATION, HOSTNAME, __SSH_PORT, USERNAME, KEY_PATH)
        results.append((load_level, stats))

    subplot_height = 4
    figure_height = subplot_height * 9
    figure_width = 15

    fig, axs = plt.subplots(9, 1, figsize=(figure_width, figure_height))

    for _, stats in results:
        if stats['avg_duration'] == 0:
            axs[0].plot(LOAD_LEVELS, [stats['avg_duration'] for _, stats in results], marker='o', color='red')
        else:
            axs[0].plot(LOAD_LEVELS, [stats['avg_duration'] for _, stats in results], marker='o')
    axs[0].set_xlabel('Количество одновременных запросов')
    axs[0].set_ylabel('Средняя длительность (с.)')
    axs[0].set_title('Средняя длительность запросов vs уровень нагрузки')
    axs[0].grid(True)

    for _, stats in results:
        if stats['avg_response_time'] == 0:
            axs[1].plot(LOAD_LEVELS, [stats['avg_response_time'] for _, stats in results], marker='o', color='red')
        else:
            axs[1].plot(LOAD_LEVELS, [stats['avg_response_time'] for _, stats in results], marker='o', color='green')
    axs[1].set_xlabel('Количество одновременных запросов')
    axs[1].set_ylabel('Время отклика (с.)')
    axs[1].set_title('Время отклика vs уровень нагрузки')
    axs[1].grid(True)

    axs[2].set_xlabel('Время')
    axs[2].set_ylabel('Средняя Загрузка CPU (%)')
    axs[2].set_title('Загрузка CPU со временем')
    for load_level, stats in results:
        axs[2].plot(stats['monitoring_timestamps'], stats['cpu_usages'], label=f'{load_level} requests')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].set_xlabel('Время')
    axs[3].set_ylabel('Средняя Загрузка CPU (Int.)')
    axs[3].set_title('Загрузка CPU со временем')
    for load_level, stats in results:
        axs[3].scatter(stats['monitoring_timestamps'], stats['cpu_usages_int'], label=f'{load_level} requests')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].set_xlabel('Длина предложения')
    axs[4].set_ylabel('Длительность запроса (с)')
    axs[4].set_title('Длительность запроса vs длина предложения')

    for load_level, stats in results:
        axs[4].scatter(stats['lengths'], stats['durations'], alpha=0.5, label=f'{load_level} requests')
        failed_mask = stats['failed']
        axs[4].scatter(
            [l for l, f in zip(stats['lengths'], failed_mask) if f],
            [d for d, f in zip(stats['durations'], failed_mask) if f],
            facecolors='none', edgecolors='r', linewidths=1.5,
        )
    axs[4].legend()
    axs[4].grid(True)

    axs[5].set_xlabel('Время')
    axs[5].set_ylabel('Средняя длительность\nобработки запроса (с)')
    axs[5].set_title('Средняя длительность обработки запросов со временем')
    for load_level, stats in results:
        request_durations_ma = moving_average(stats['durations'], 5)
        request_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in stats['timestamps']]

        if len(request_timestamps) < len(request_durations_ma):
            continue

        trimmed_timestamps = request_timestamps[-len(request_durations_ma):]
        if all(request_durations_ma) == 0:
            axs[5].plot(trimmed_timestamps, request_durations_ma, label=f'{load_level} requests', color='red')
        else:
            axs[5].plot(trimmed_timestamps, request_durations_ma, label=f'{load_level} requests')
    axs[5].legend()
    axs[5].grid(True)

    axs[6].set_xlabel('Время')
    axs[6].set_ylabel('Длительность запроса (с)')
    axs[6].set_title('Длительность запроса по времени')
    for load_level, stats in results:
        request_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in stats['timestamps']]
        failed_mask = stats['failed']
        axs[6].scatter(request_timestamps, stats['durations'], alpha=0.5, label=f'{load_level} requests')
        axs[6].scatter(
            [ts for ts, f in zip(request_timestamps, failed_mask) if f],
            [d for d, f in zip(stats['durations'], failed_mask) if f],
            facecolors='none', edgecolors='r', linewidths=1.5,
        )
    axs[6].legend()
    axs[6].grid(True)

    axs[7].set_xlabel('Время')
    axs[7].set_ylabel('Использование памяти (Mb)')
    axs[7].set_title('Использование памяти со временем')
    for load_level, stats in results:
        axs[7].plot(stats['monitoring_timestamps'], stats['memory_usages'], label=f'{load_level} requests')
    axs[7].yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axs[7].yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))
    axs[7].legend()
    axs[7].grid(True)

    axs[8].set_xlabel('Длина запроса (символы)')
    axs[8].set_ylabel('Количество одновременных запросов')
    axs[8].set_title('Статистика запросов')

    for load_level, stats in results:
        failed_mask = stats['failed']

        axs[8].scatter(
            [l for l, f in zip(stats['lengths'], failed_mask) if not f],
            [load_level] * sum(not f for f in failed_mask),
            color='blue', alpha=0.5, label='Success' if load_level == LOAD_LEVELS[0] else ""
        )

        axs[8].scatter(
            [l for l, f in zip(stats['lengths'], failed_mask) if f],
            [load_level] * sum(failed_mask),
            color='red', marker='x', s=50, label='Failed' if load_level == LOAD_LEVELS[0] else ""
        )

    axs[8].legend()
    axs[8].grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('test_results.png', dpi=200, bbox_inches='tight')

    show_statistics(results)


if __name__ == '__main__':
    __main()
