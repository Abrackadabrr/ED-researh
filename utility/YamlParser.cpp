#include "YamlParser.hpp"

#include <stdexcept>
#include <string>

namespace Utility::Yaml {

YAML::Node parse_file(const std::filesystem::path &path) {
    if (path.empty()) {
        throw std::invalid_argument("YAML configuration path must not be empty");
    }

    try {
        YAML::Node document = YAML::LoadFile(path.string());
        if (!document || document.IsNull()) {
            throw std::runtime_error("YAML document is empty: " + path.string());
        }
        return document;
    } catch (const YAML::Exception &error) {
        throw std::runtime_error("Cannot parse YAML file '" + path.string() + "': " + error.what());
    }
}

} // namespace Utility::Yaml
