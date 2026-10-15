// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026

#include "dispatch_plan.hpp"
#include "kernel_launch.hpp"

#include <acl/acl.h>
#include <shmem.h>
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace deepep::dispatch_bench {
namespace {

constexpr uint64_t kChunkBytes = 4ULL * 1024ULL * 1024ULL;
constexpr double kCyclesPerUs = 1000.0;
constexpr uint32_t kCrossLaunchGroups = 28U;
constexpr std::array<const char*, kTrafficKinds> kTrafficNames{"local", "group", "cross"};

struct Options {
    Config config;
    uint32_t rank = 0U;
    uint32_t device = 0U;
    uint32_t warmup = 10U;
    uint32_t iterations = 20U;
    uint32_t repeats = 1U;
    MetadataMode metadata_mode = MetadataMode::Dma;
    std::string endpoint;
    std::filesystem::path output;
    bool timer_requested = false;
    bool help = false;
};

struct Round {
    double kernel_us;
    double envelope_us;
    std::vector<TimerRecord> records;
};

const char* MetadataModeName(MetadataMode mode)
{
    switch (mode) {
        case MetadataMode::Scalar:
            return "scalar";
        case MetadataMode::Dma:
            return "dma";
        case MetadataMode::Preload:
            return "preload";
    }
    throw std::invalid_argument("Invalid metadata mode");
}

void Check(int result, const char* operation)
{
    if (result != 0) {
        throw std::runtime_error(std::string(operation) + " failed (status=" + std::to_string(result) + ")");
    }
}

uint32_t ParseUint(const std::string& value, const std::string& name)
{
    if (value.empty()) {
        throw std::invalid_argument(name + " requires an unsigned integer");
    }
    uint64_t number = 0U;
    for (char digit : value) {
        if (digit < '0' || digit > '9') {
            throw std::invalid_argument(name + " requires an unsigned integer");
        }
        number = number * 10U + static_cast<uint32_t>(digit - '0');
        if (number > UINT32_MAX) {
            throw std::invalid_argument(name + " exceeds uint32 capacity");
        }
    }
    return static_cast<uint32_t>(number);
}

std::array<uint32_t, kTrafficKinds> ParseWorkers(const std::string& value)
{
    std::array<uint32_t, kTrafficKinds> counts{};
    std::size_t begin = 0U;
    for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
        const std::size_t comma = value.find(',', begin);
        if ((traffic + 1U < kTrafficKinds) != (comma != std::string::npos)) {
            throw std::invalid_argument("--workers requires exactly local,group,cross counts");
        }
        counts[traffic] =
            ParseUint(value.substr(begin, comma == std::string::npos ? comma : comma - begin), "--workers");
        begin = comma == std::string::npos ? value.size() : comma + 1U;
    }
    return counts;
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    std::unordered_set<std::string> seen;
    for (int index = 1; index < argc; ++index) {
        const std::string name(argv[index]);
        if (name == "--help") {
            options.help = true;
            return options;
        }
        if (!seen.insert(name).second) {
            throw std::invalid_argument("Duplicate argument: " + name);
        }
        if (name == "--timer") {
            options.timer_requested = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + name);
        }
        const std::string value(argv[++index]);
        if (name == "--rank") {
            options.rank = ParseUint(value, name);
        } else if (name == "--world") {
            options.config.world = ParseUint(value, name);
        } else if (name == "--device") {
            options.device = ParseUint(value, name);
        } else if (name == "--endpoint") {
            options.endpoint = value;
        } else if (name == "--output") {
            options.output = value;
        } else if (name == "--bs") {
            options.config.batch = ParseUint(value, name);
        } else if (name == "--hidden") {
            options.config.hidden = ParseUint(value, name);
        } else if (name == "--intermediate") {
            options.config.intermediate = ParseUint(value, name);
        } else if (name == "--experts") {
            options.config.experts = ParseUint(value, name);
        } else if (name == "--topk") {
            options.config.topk = ParseUint(value, name);
        } else if (name == "--group-size") {
            options.config.group_size = ParseUint(value, name);
        } else if (name == "--cores") {
            options.config.cores = ParseUint(value, name);
        } else if (name == "--workers") {
            options.config.worker_counts = ParseWorkers(value);
        } else if (name == "--buffer-count") {
            options.config.ring_slots = ParseUint(value, name);
        } else if (name == "--repeat") {
            options.repeats = ParseUint(value, name);
        } else if (name == "--warmup") {
            options.warmup = ParseUint(value, name);
        } else if (name == "--iters") {
            options.iterations = ParseUint(value, name);
        } else if (name == "--metadata-mode") {
            if (value != "scalar" && value != "dma" && value != "preload") {
                throw std::invalid_argument("--metadata-mode must be scalar, dma, or preload");
            }
            options.metadata_mode = value == "scalar" ? MetadataMode::Scalar
                                    : value == "dma"  ? MetadataMode::Dma
                                                      : MetadataMode::Preload;
        } else if (name == "--traffic") {
            if (value != "all" && value != "cross") {
                throw std::invalid_argument("--traffic must be all or cross");
            }
            options.config.cross_only = value == "cross";
        } else if (name == "--scope") {
            if (value != "all" && value != "first-expert") {
                throw std::invalid_argument("--scope must be all or first-expert");
            }
            options.config.first_expert = value == "first-expert";
        } else {
            throw std::invalid_argument("Unknown argument: " + name);
        }
    }
    if (options.config.cross_only && seen.count("--workers") == 0U) {
        options.config.worker_counts = {0U, 0U, options.config.cores};
    }
    for (const char* name : {"--rank", "--device", "--endpoint", "--output"}) {
        if (seen.count(name) == 0U) {
            throw std::invalid_argument(std::string("Missing required argument: ") + name);
        }
    }
    if (options.output.empty() || options.endpoint.rfind("tcp://", 0U) != 0U) {
        throw std::invalid_argument("--output must be nonempty and --endpoint must start with tcp://");
    }
    if (options.device > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) || options.repeats == 0U ||
        options.repeats > 65536U || options.iterations == 0U || options.iterations > 100000U ||
        options.warmup > 100000U) {
        throw std::invalid_argument(
            "device exceeds int32; repeat must be [1,65536], iters [1,100000], warmup [0,100000]");
    }
    return options;
}

uint8_t* Allocate(uint64_t bytes, std::vector<void*>& allocations)
{
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Invalid local GM allocation size");
    }
    void* pointer = nullptr;
    Check(aclrtMalloc(&pointer, static_cast<std::size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    allocations.push_back(pointer);
    return static_cast<uint8_t*>(pointer);
}

void CopyToDevice(void* destination, const void* source, uint64_t bytes)
{
    Check(aclrtMemcpy(destination, static_cast<std::size_t>(bytes), source, static_cast<std::size_t>(bytes),
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "aclrtMemcpy(H2D)");
}

void CopyToHost(void* destination, const void* source, uint64_t bytes)
{
    Check(aclrtMemcpy(destination, static_cast<std::size_t>(bytes), source, static_cast<std::size_t>(bytes),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy(D2H)");
}

uint32_t InputWord(uint32_t rank, uint32_t token, uint32_t word)
{
    return (rank << 28U) | (token << 12U) | word;
}

void InitializeSource(uint8_t* source, const Options& options, const Plan& plan)
{
    if (plan.row_bytes % sizeof(uint32_t) != 0U || plan.row_bytes / sizeof(uint32_t) > 4096U) {
        throw std::invalid_argument("Source rows do not fit the rank/token/word input encoding");
    }
    const uint32_t words_per_row = plan.row_bytes / sizeof(uint32_t);
    const uint32_t rows_per_chunk = static_cast<uint32_t>(std::max(uint64_t{1U}, kChunkBytes / plan.row_bytes));
    std::vector<uint32_t> input(static_cast<std::size_t>(rows_per_chunk) * words_per_row);
    for (uint32_t begin = 0U; begin < options.config.batch;) {
        const uint32_t rows = std::min(rows_per_chunk, options.config.batch - begin);
        for (uint32_t row = 0U; row < rows; ++row) {
            for (uint32_t word = 0U; word < words_per_row; ++word) {
                input[static_cast<std::size_t>(row) * words_per_row + word] =
                    InputWord(options.rank, begin + row, word);
            }
        }
        CopyToDevice(source + static_cast<uint64_t>(begin) * plan.row_bytes, input.data(),
                     static_cast<uint64_t>(rows) * plan.row_bytes);
        begin += rows;
    }
}

void InitializeMetadata(uint8_t* metadata_input, const Plan& plan)
{
    constexpr uint32_t record_bytes = kMetaWords * sizeof(uint32_t);
    constexpr uint32_t rows_per_chunk = static_cast<uint32_t>(kChunkBytes / record_bytes);
    // Keep a bounded host staging buffer; the table is indexed by task order,
    // independently of each task's final output destination.
    std::vector<uint32_t> input(static_cast<std::size_t>(rows_per_chunk) * kMetaWords, 0U);
    for (uint32_t begin = 0U; begin < plan.total_rows;) {
        const uint32_t rows = std::min(rows_per_chunk, plan.total_rows - begin);
        for (uint32_t row = 0U; row < rows; ++row) {
            const CopyTask& task = plan.tasks[static_cast<std::size_t>(begin) + row];
            const std::size_t offset = static_cast<std::size_t>(row) * kMetaWords;
            input[offset] = task.peer;
            input[offset + 1U] = task.token;
            input[offset + 2U] = task.topk_slot;
        }
        CopyToDevice(metadata_input + static_cast<uint64_t>(begin) * record_bytes, input.data(),
                     static_cast<uint64_t>(rows) * record_bytes);
        begin += rows;
    }
}

void CheckRowBytes(const uint8_t* actual, uint32_t length, const CopyTask& task, uint32_t source_offset,
                   uint32_t destination, const char* label)
{
    uint32_t offset = 0U;
    for (; offset + sizeof(uint32_t) <= length; offset += sizeof(uint32_t)) {
        uint32_t observed = 0U;
        std::memcpy(&observed, actual + offset, sizeof(observed));
        const uint32_t expected = InputWord(task.peer, task.token, (source_offset + offset) / sizeof(uint32_t));
        if (observed != expected) {
            throw std::runtime_error(std::string("Output ") + label + " mismatch: destination=" +
                                     std::to_string(destination) + ", peer=" + std::to_string(task.peer) +
                                     ", token=" + std::to_string(task.token) + ", byte=" + std::to_string(offset));
        }
    }
    if (offset < length) {
        const uint32_t expected = InputWord(task.peer, task.token, (source_offset + offset) / sizeof(uint32_t));
        if (std::memcmp(actual + offset, &expected, length - offset) != 0) {
            throw std::runtime_error(std::string("Output ") + label +
                                     " tail mismatch: destination=" + std::to_string(destination));
        }
    }
}

void ValidateOutput(const Options& options, const Plan& plan, const uint8_t* output_tokens,
                    const uint8_t* output_scales, const uint8_t* output_metadata)
{
    std::vector<uint32_t> task_at_destination(plan.total_rows, UINT32_MAX);
    for (uint32_t index = 0U; index < plan.tasks.size(); ++index) {
        const uint32_t destination = plan.tasks[index].destination;
        if (destination >= plan.total_rows || task_at_destination[destination] != UINT32_MAX) {
            throw std::logic_error("Dispatch plan has duplicate/out-of-range destinations");
        }
        task_at_destination[destination] = index;
    }
    const uint32_t rows_per_chunk = static_cast<uint32_t>(std::max(uint64_t{1U}, kChunkBytes / options.config.hidden));
    std::vector<uint8_t> tokens(static_cast<std::size_t>(rows_per_chunk) * options.config.hidden);
    std::vector<uint8_t> scales(static_cast<std::size_t>(rows_per_chunk) * plan.scale_elements);
    std::vector<uint32_t> metadata(static_cast<std::size_t>(rows_per_chunk) * kMetaWords);
    const uint32_t scale_offset = (options.config.hidden + 255U) / 256U * 256U;
    for (uint32_t begin = 0U; begin < plan.total_rows;) {
        const uint32_t rows = std::min(rows_per_chunk, plan.total_rows - begin);
        CopyToHost(tokens.data(), output_tokens + static_cast<uint64_t>(begin) * options.config.hidden,
                   static_cast<uint64_t>(rows) * options.config.hidden);
        CopyToHost(scales.data(), output_scales + static_cast<uint64_t>(begin) * plan.scale_elements,
                   static_cast<uint64_t>(rows) * plan.scale_elements);
        CopyToHost(metadata.data(), output_metadata + static_cast<uint64_t>(begin) * kMetaWords * sizeof(uint32_t),
                   static_cast<uint64_t>(rows) * kMetaWords * sizeof(uint32_t));
        for (uint32_t row = 0U; row < rows; ++row) {
            const uint32_t destination = begin + row;
            if (task_at_destination[destination] == UINT32_MAX) {
                throw std::logic_error("Dispatch plan omitted an output row");
            }
            const CopyTask& task = plan.tasks[task_at_destination[destination]];
            CheckRowBytes(tokens.data() + static_cast<std::size_t>(row) * options.config.hidden, options.config.hidden,
                          task, 0U, destination, "token");
            CheckRowBytes(scales.data() + static_cast<std::size_t>(row) * plan.scale_elements, plan.scale_elements,
                          task, scale_offset, destination, "scale");
            const std::array<uint32_t, kMetaWords> expected_metadata{task.peer, task.token, task.topk_slot, 0U, 0U, 0U,
                                                                     0U,        0U};
            for (uint32_t word = 0U; word < kMetaWords; ++word) {
                if (metadata[static_cast<std::size_t>(row) * kMetaWords + word] != expected_metadata[word]) {
                    throw std::runtime_error("Output metadata mismatch: destination=" + std::to_string(destination) +
                                             ", word=" + std::to_string(word));
                }
            }
        }
        begin += rows;
    }
}

double ValidateTimers(const Options& options, const Plan& plan, const std::vector<TimerRecord>& records)
{
    uint64_t first = UINT64_MAX;
    uint64_t last = 0U;
    for (uint32_t worker = 0U; worker < options.config.cores; ++worker) {
        const TimerRecord& record = records[worker];
        const Worker& expected = plan.workers[worker];
        const uint64_t rows = static_cast<uint64_t>(expected.count) * options.repeats;
        if (record.traffic != expected.traffic || record.rows != rows || record.end < record.start ||
            (rows != 0U && (record.start == 0U || record.end == record.start))) {
            throw std::runtime_error("Invalid device timer record for group " + std::to_string(worker));
        }
        if (rows != 0U) {
            first = std::min(first, record.start);
            last = std::max(last, record.end);
        }
    }
    if (first == UINT64_MAX || last <= first) {
        throw std::runtime_error("No active device timer intervals were recorded");
    }
    return static_cast<double>(last - first) / kCyclesPerUs;
}

std::string Report(const Options& options, const Plan& plan, const std::vector<Round>& rounds)
{
    std::ostringstream json;
    json << std::setprecision(17) << "{\n\"schema_version\":3,\"rank\":" << options.rank
         << ",\"world\":" << options.config.world << ",\"device\":" << options.device
         << ",\"cores\":" << options.config.cores << ",\"group_size\":" << options.config.group_size
         << ",\"batch\":" << options.config.batch << ",\"hidden\":" << options.config.hidden
         << ",\"intermediate\":" << options.config.intermediate << ",\"experts\":" << options.config.experts
         << ",\"topk\":" << options.config.topk << ",\"scope\":\""
         << (options.config.first_expert ? "first-expert" : "all") << "\",\"metadata_mode\":\""
         << MetadataModeName(options.metadata_mode) << "\",\"traffic_selection\":\""
         << (options.config.cross_only ? "cross" : "all") << "\",\"row_bytes\":" << plan.row_bytes
         << ",\"scale_elements\":" << plan.scale_elements << ",\"buffer_count\":" << options.config.ring_slots
         << ",\"repeats\":" << options.repeats
         << ",\"cycles_per_us\":1000.0,\"timer_requested\":" << (options.timer_requested ? "true" : "false")
         << ",\"traffic_rows\":{";
    for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
        json << (traffic == 0U ? "" : ",") << '"' << kTrafficNames[traffic] << "\":" << plan.traffic_rows[traffic];
    }
    json << "},\"worker_counts\":{";
    for (uint32_t traffic = 0U; traffic < kTrafficKinds; ++traffic) {
        json << (traffic == 0U ? "" : ",") << '"' << kTrafficNames[traffic]
             << "\":" << options.config.worker_counts[traffic];
    }
    json << "},\"validation\":{\"passed\":true,\"checked_rows\":" << plan.total_rows << "},\"rounds\":[\n";
    for (std::size_t iteration = 0U; iteration < rounds.size(); ++iteration) {
        const Round& round = rounds[iteration];
        json << (iteration == 0U ? "" : ",\n") << "{\"iteration\":" << iteration
             << ",\"kernel_elapsed_us\":" << round.kernel_us << ",\"timer_envelope_us\":" << round.envelope_us
             << ",\"workers\":[";
        for (uint32_t worker = 0U; worker < options.config.cores; ++worker) {
            const TimerRecord& record = round.records[worker];
            json << (worker == 0U ? "" : ",") << "{\"group_id\":" << worker << ",\"traffic\":\""
                 << kTrafficNames[plan.workers[worker].traffic] << "\",\"rows\":" << record.rows
                 << ",\"start_cycle\":" << record.start << ",\"end_cycle\":" << record.end << '}';
        }
        json << "]}";
    }
    json << "\n]}\n";
    return json.str();
}

void WriteReport(const std::filesystem::path& destination, const std::string& report)
{
    std::filesystem::path temporary = destination;
    temporary += ".tmp." + std::to_string(static_cast<uint64_t>(getpid()));
    try {
        std::ofstream stream(temporary, std::ios::out | std::ios::binary | std::ios::trunc);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << report;
        stream.close();
        std::filesystem::rename(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void Run(const Options& options)
{
    const Plan plan = BuildPlan(options.config, options.rank);
    if (plan.total_rows == 0U) {
        throw std::invalid_argument("Selected traffic has no rows; increase --bs or --topk");
    }
    // Keep the original 28-group launch shape throughout the CROSS sweep.
    // config.cores still counts active AIV1 workers; the other groups return
    // before accessing their worker or timer buffers.
    const uint32_t launch_groups = options.config.cross_only ? kCrossLaunchGroups : options.config.cores;
    if (options.config.cores > launch_groups) {
        throw std::invalid_argument("CROSS-only traffic supports at most 28 active workers in its fixed launch");
    }
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(static_cast<int32_t>(options.device)), "aclrtSetDevice");
    const char* soc = aclrtGetSocName();
    if (soc == nullptr || std::string(soc).rfind("Ascend950", 0U) != 0U) {
        throw std::runtime_error("This dav-3510 benchmark and its timer conversion are qualified only for Ascend950");
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        throw std::runtime_error("Failed to query the Ascend platform");
    }
    uint64_t ub_bytes = 0U;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    uint64_t metadata_ub_bytes = static_cast<uint64_t>(options.config.ring_slots) * kMetaWords * sizeof(uint32_t);
    if (options.metadata_mode == MetadataMode::Preload) {
        uint32_t max_worker_rows = 1U;
        for (const Worker& worker : plan.workers) {
            max_worker_rows = std::max(max_worker_rows, worker.count);
        }
        metadata_ub_bytes = static_cast<uint64_t>(max_worker_rows) * kMetaWords * sizeof(uint32_t);
    }
    const uint64_t required_ub =
        static_cast<uint64_t>(options.config.ring_slots) * plan.row_bytes + metadata_ub_bytes + 4096U;
    if (launch_groups > aic || static_cast<uint64_t>(launch_groups) * 2U > aiv) {
        throw std::runtime_error(
            "Requested launch groups exceed the available AIC/AIV resources; launch will not shrink");
    }
    if (ub_bytes < required_ub || ub_bytes > UINT32_MAX) {
        if (options.metadata_mode == MetadataMode::Preload) {
            throw std::runtime_error(
                "Preloaded worker metadata and Dispatch row ring exceed the platform UB budget; "
                "use --metadata-mode dma or reduce the workload");
        }
        throw std::runtime_error("Dispatch row ring and metadata do not fit the platform UB budget");
    }
    std::cerr << "[rank " << options.rank << "] launch_groups=" << launch_groups
              << " active_workers=" << options.config.cores << " idle_groups=" << launch_groups - options.config.cores
              << '\n';
    const uint64_t source_bytes = static_cast<uint64_t>(options.config.batch) * plan.row_bytes;
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t unique_id{};
    if (options.endpoint.size() >= sizeof(attr.ip_port)) {
        throw std::invalid_argument("SHMEM endpoint is too long");
    }
    attr.my_pe = static_cast<int>(options.rank);
    attr.n_pes = static_cast<int>(options.config.world);
    attr.local_mem_size = detail::AlignUp(source_bytes + 2ULL * 1024ULL * 1024ULL, 2ULL * 1024ULL * 1024ULL);
    std::memcpy(attr.ip_port, options.endpoint.c_str(), options.endpoint.size() + 1U);
    attr.comm_args = &unique_id;
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    attr.option_attr.shm_init_timeout = 120U;
    attr.option_attr.shm_create_timeout = 120U;
    attr.option_attr.control_operation_timeout = 120U;
    Check(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr");
    uint8_t* source = static_cast<uint8_t*>(aclshmem_malloc(static_cast<std::size_t>(source_bytes)));
    if (source == nullptr) {
        throw std::runtime_error("SHMEM packed source allocation failed");
    }
    std::vector<uint64_t> peers(options.config.world);
    for (uint32_t peer = 0U; peer < options.config.world; ++peer) {
        void* pointer = aclshmem_ptr(source, static_cast<int>(peer));
        if (pointer == nullptr) {
            throw std::runtime_error("Peer " + std::to_string(peer) +
                                     " cannot be mapped by aclshmem_ptr; this benchmark supports intra-node MTE only");
        }
        peers[peer] = reinterpret_cast<uintptr_t>(pointer);
    }
    InitializeSource(source, options, plan);
    std::vector<void*> allocations;
    uint8_t* peer_table = Allocate(peers.size() * sizeof(uint64_t), allocations);
    uint8_t* tasks = Allocate(plan.tasks.size() * sizeof(CopyTask), allocations);
    uint8_t* workers = Allocate(plan.workers.size() * sizeof(Worker), allocations);
    uint8_t* timer = Allocate(plan.workers.size() * sizeof(TimerRecord), allocations);
    const uint64_t token_bytes = static_cast<uint64_t>(plan.total_rows) * options.config.hidden;
    const uint64_t scale_bytes = static_cast<uint64_t>(plan.total_rows) * plan.scale_elements;
    const uint64_t metadata_bytes = static_cast<uint64_t>(plan.total_rows) * kMetaWords * sizeof(uint32_t);
    uint8_t* metadata_input = nullptr;
    if (options.metadata_mode != MetadataMode::Scalar) {
        metadata_input = Allocate(metadata_bytes, allocations);
        InitializeMetadata(metadata_input, plan);
    }
    uint8_t* output_tokens = Allocate(token_bytes, allocations);
    uint8_t* output_scales = Allocate(scale_bytes, allocations);
    uint8_t* output_metadata = Allocate(metadata_bytes, allocations);
    CopyToDevice(peer_table, peers.data(), peers.size() * sizeof(uint64_t));
    CopyToDevice(tasks, plan.tasks.data(), plan.tasks.size() * sizeof(CopyTask));
    CopyToDevice(workers, plan.workers.data(), plan.workers.size() * sizeof(Worker));
    aclrtStream stream = nullptr;
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    Check(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
    Check(aclrtCreateEvent(&end), "aclrtCreateEvent(end)");
    KernelConfig kernel{options.rank,
                        options.config.world,
                        options.config.cores,
                        options.config.hidden,
                        plan.scale_elements,
                        plan.row_bytes,
                        options.config.ring_slots,
                        options.repeats,
                        static_cast<uint32_t>(options.metadata_mode)};
    std::vector<Round> rounds;
    rounds.reserve(options.iterations);
    const uint32_t total_rounds = options.warmup + options.iterations;
    std::cerr << "[rank " << options.rank << "] source_stride=" << plan.row_bytes << " rows=" << plan.total_rows
              << " workers=" << options.config.worker_counts[0] << ',' << options.config.worker_counts[1] << ','
              << options.config.worker_counts[2] << " metadata_mode=" << MetadataModeName(options.metadata_mode)
              << " traffic=" << (options.config.cross_only ? "cross" : "all") << " AIC/AIV=" << aic << '/' << aiv
              << '\n';
    for (uint32_t iteration = 0U; iteration < total_rounds; ++iteration) {
        // These resets and the collective are deliberately outside ACL timing.
        // Every measured round must produce its own output, not inherit warmup data.
        Check(aclrtMemset(output_tokens, static_cast<std::size_t>(token_bytes), 0xA5,
                          static_cast<std::size_t>(token_bytes)),
              "aclrtMemset(output_tokens)");
        Check(aclrtMemset(output_scales, static_cast<std::size_t>(scale_bytes), 0xA5,
                          static_cast<std::size_t>(scale_bytes)),
              "aclrtMemset(output_scales)");
        Check(aclrtMemset(output_metadata, static_cast<std::size_t>(metadata_bytes), 0xA5,
                          static_cast<std::size_t>(metadata_bytes)),
              "aclrtMemset(output_metadata)");
        Check(
            aclrtMemset(timer, plan.workers.size() * sizeof(TimerRecord), 0, plan.workers.size() * sizeof(TimerRecord)),
            "aclrtMemset(timer)");
        aclshmem_barrier_all();
        Check(aclrtRecordEvent(start, stream), "aclrtRecordEvent(start)");
        dispatch_bandwidth_kernel_do(launch_groups, static_cast<uint32_t>(ub_bytes), stream, peer_table, tasks,
                                     metadata_input, workers, output_tokens, output_scales, output_metadata, timer,
                                     kernel);
        Check(aclrtRecordEvent(end, stream), "aclrtRecordEvent(end)");
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        if (iteration >= options.warmup) {
            float elapsed_ms = 0.0F;
            Check(aclrtEventElapsedTime(&elapsed_ms, start, end), "aclrtEventElapsedTime");
            const double elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
            if (!std::isfinite(elapsed_us) || elapsed_us <= 0.0) {
                throw std::runtime_error("ACL event timing must be finite and positive");
            }
            std::vector<TimerRecord> records(plan.workers.size());
            CopyToHost(records.data(), timer, records.size() * sizeof(TimerRecord));
            const double envelope = ValidateTimers(options, plan, records);
            rounds.push_back({elapsed_us, envelope, std::move(records)});
        }
    }
    ValidateOutput(options, plan, output_tokens, output_scales, output_metadata);
    const std::string report = Report(options, plan, rounds);

    // Normal cleanup is collective. No destructor/catch handler enters SHMEM:
    // an error must let the launcher stop the remaining ranks instead of hanging.
    aclshmem_barrier_all();
    aclshmem_free(source);
    for (void* allocation : allocations) {
        Check(aclrtFree(allocation), "aclrtFree");
    }
    Check(aclshmem_finalize(), "aclshmem_finalize");
    Check(aclrtDestroyEvent(end), "aclrtDestroyEvent(end)");
    Check(aclrtDestroyEvent(start), "aclrtDestroyEvent(start)");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtResetDevice(static_cast<int32_t>(options.device)), "aclrtResetDevice");
    Check(aclFinalize(), "aclFinalize");
    WriteReport(options.output, report);
    std::cerr << "[rank " << options.rank << "] validated " << plan.total_rows << " rows; wrote " << options.output
              << '\n';
}

}  // namespace
}  // namespace deepep::dispatch_bench

int main(int argc, char** argv)
{
    try {
        const auto options = deepep::dispatch_bench::ParseOptions(argc, argv);
        if (options.help) {
            std::cout << "Dispatch MTE benchmark: --rank R --device D --endpoint tcp://IP:PORT --output FILE\n"
                         "[--world 16 --bs 4096 --hidden 3584 --intermediate 1280 --experts 384 --topk 8]\n"
                         "[--group-size 4 --cores 28 --workers 2,5,21 --buffer-count 6 --repeat 1]\n"
                         "[--traffic all|cross --scope all|first-expert --metadata-mode scalar|dma|preload]\n"
                         "[--workers defaults: all=2,5,21; cross=0,0,cores]\n"
                         "[cross: fixed 28 launch groups, cores=1..28 active AIV1 workers]\n"
                         "[--warmup 10 --iters 20 --timer]\n";
            return 0;
        }
        deepep::dispatch_bench::Run(options);
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "dispatch_bandwidth: " << failure.what() << '\n';
        return 1;
    }
}
