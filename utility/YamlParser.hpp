#pragma once

#include <filesystem>

#include <yaml-cpp/yaml.h>

namespace Utility::Yaml {

// Loads a YAML document into yaml-cpp's standard tree representation.
// YAML::Exception is wrapped into std::runtime_error with the source path.
[[nodiscard]] YAML::Node parse_file(const std::filesystem::path &path);

} // namespace Utility::Yaml
