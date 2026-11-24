#include <algorithmfactory.h>
#include <essentia.h>

#include <bit>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
using essentia::Real;
using essentia::standard::Algorithm;
using essentia::standard::AlgorithmFactory;

void write_vector(const std::vector<Real>& values) {
    static_assert(sizeof(Real) == sizeof(float));
    static_assert(std::endian::native == std::endian::little);
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Essentia output exceeds the sample limit");
    }
    const auto count = static_cast<std::uint32_t>(values.size());
    std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
    std::cout.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(Real));
}

void analyze(const char* filename) {
    constexpr int sample_rate = 44100;
    auto& factory = AlgorithmFactory::instance();
    std::vector<Real> samples, pitch, confidence;
    const std::unique_ptr<Algorithm> loader(factory.create(
        "MonoLoader", "filename", filename, "sampleRate", sample_rate
    ));
    loader->output("audio").set(samples);
    loader->compute();
    if (!samples.empty()) {
        const std::unique_ptr<Algorithm> detector(factory.create(
            "PredominantPitchMelodia", "sampleRate", sample_rate,
            "frameSize", 2048, "hopSize", 256,
            "minFrequency", 55.0, "maxFrequency", 1760.0, "voicingTolerance", 0.2
        ));
        detector->input("signal").set(samples);
        detector->output("pitch").set(pitch);
        detector->output("pitchConfidence").set(confidence);
        detector->compute();
    }
    write_vector(samples);
    write_vector(pitch);
    write_vector(confidence);
    if (!std::cout) {
        throw std::runtime_error("Could not write MELODIA frame data");
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: karaoke-melodia audio-file\n";
        return 2;
    }
#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        std::cerr << "Could not set binary output mode\n";
        return 1;
    }
#endif
    essentia::init();
    int status = 0;
    try {
        analyze(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    essentia::shutdown();
    return status;
}
