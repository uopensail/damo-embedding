#include "argparse.hpp"
#include "embedding.h"
#include "httplib.h"

#pragma pack(push)
#pragma pack(1)

struct PullRequest {
  int group;
  int size;
  int64_t keys[];
};

struct PushRequest {
  int group;
  int size;
  char data[];
};

#pragma pack(pop)

int main(int argc, char const *argv[]) {
  auto args = util::argparser("damo embedding server");
  args.set_program_name("damo-server")
      .add_help_option()
      .add_option<std::string>(
          "-c", "--config",
          "configure file, json format, default: ./configure.json",
          "./configure.json")
      .parse(argc, argv);

  std::string config_path = args.get_option<std::string>("--config");

  std::shared_ptr<EmbeddingWareHouse> warehouse =
      std::make_shared<EmbeddingWareHouse>(config_path);

  httplib::Server srv;
  srv.Post("/pull", [=](const httplib::Request &req, httplib::Response &res) {
    PullRequest *ptr = (PullRequest *)(req.body.data());
    int dim = warehouse->dim(ptr->group);
    int n = dim * ptr->size;
    std::string result("", n * sizeof(Float));
    Float *data = (Float *)(result.data());
    warehouse->lookup(ptr->group, ptr->keys, ptr->size, data, n);
    res.set_content(result, "application/octet-stream");
  });

  srv.Post("/push", [=](const httplib::Request &req, httplib::Response &res) {
    PushRequest *ptr = (PushRequest *)(req.body.data());
    int dim = warehouse->dim(ptr->group);
    int n = dim * ptr->size;
    int64_t *keys = (int64_t *)ptr->data;
    Float *gds = (Float *)&(ptr->data[sizeof(int64_t) * ptr->size]);
    warehouse->lookup(ptr->group, keys, ptr->size, gds, n);
    res.status = 200;
    return;
  });

  //   srv.Get("/dump", [=](const httplib::Request &req, httplib::Response &res)
  //   {
  //     int64_t ts = get_current_time();
  //     std::string path = "/tmp/ds-dump-" + std::to_string(ts) + ".dat";
  //     warehouse->dump(path);
  //     res.set_header("Cache-Control", "no-cache");
  //     res.set_header("Content-Disposition", "attachment;
  //     filename=sparse.dat"); std::ifstream reader(path, std::ifstream::binary
  //     | std::ifstream::in); reader.seekg(0, reader.end); size_t size =
  //     reader.tellg(); reader.close();

  //     res.set_content_provider(
  //         size, "application/octet-stream",
  //         [size, path](size_t offset, size_t length, httplib::DataSink &sink)
  //         {
  //           std::ifstream reader(path, std::ifstream::binary |
  //           std::ifstream::in); if (!reader.good()) {
  //             return false;
  //           }
  //           if (offset >= size) {
  //             return false;
  //           }
  //           const size_t chunk = 1024 * 1024;
  //           size_t read_size = std::min<size_t>(chunk, size - offset);
  //           char buffer[chunk];
  //           reader.seekg(offset, reader.beg);
  //           reader.read(buffer, read_size);
  //           reader.close();
  //           sink.write(buffer, read_size);
  //           return true;
  //         });
  //     std::filesystem::remove_all(path);
  //   });

  srv.Post("/dump", [=](const httplib::Request &req, httplib::Response &res) {
    std::string path = req.body;
    warehouse->dump(path);
    res.status = 200;
  });

  srv.Get("/stop", [&](const httplib::Request &req, httplib::Response &res) {
    srv.stop();
  });

  //   srv.Get("/checkpoint", [=](const httplib::Request &req,
  //                              httplib::Response &res) {
  //     int64_t ts = get_current_time();
  //     std::string path = "/tmp/ds-checkpoint-" + std::to_string(ts) + ".dat";
  //     warehouse->checkpoint(path);
  //     res.set_header("Cache-Control", "no-cache");
  //     res.set_header("Content-Disposition",
  //                    "attachment; filename=checkpoint.dat");
  //     std::ifstream reader(path, std::ifstream::binary | std::ifstream::in);
  //     reader.seekg(0, reader.end);
  //     size_t size = reader.tellg();
  //     reader.close();

  //     res.set_content_provider(
  //         size, "application/octet-stream",
  //         [size, path](size_t offset, size_t length, httplib::DataSink &sink)
  //         {
  //           std::ifstream reader(path, std::ifstream::binary |
  //           std::ifstream::in); if (!reader.good()) {
  //             return false;
  //           }
  //           if (offset >= size) {
  //             return false;
  //           }
  //           const size_t chunk = 1024 * 1024;
  //           size_t read_size = std::min<size_t>(chunk, size - offset);
  //           char buffer[chunk];
  //           reader.seekg(offset, reader.beg);
  //           reader.read(buffer, read_size);
  //           reader.close();
  //           sink.write(buffer, read_size);
  //           return true;
  //         });
  //     std::filesystem::remove_all(path);
  //   });

  srv.Post("/checkpoint",
           [=](const httplib::Request &req, httplib::Response &res) {
             std::string path = req.body;
             warehouse->checkpoint(path);
             res.status = 200;
           });

  srv.listen("0.0.0.0", 9275);
  return 0;
}