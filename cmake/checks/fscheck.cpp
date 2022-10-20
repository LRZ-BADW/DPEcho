#include <string>
#include <filesystem>

int main(int argc, const char **argv) {
  std::string dir = "bla!";
  std::filesystem::create_directory(dir);
}
