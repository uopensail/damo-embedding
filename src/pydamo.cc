#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "damo.h"

namespace py = pybind11;

PYBIND11_MODULE(damo, m) {
  m.doc() = R"pbdoc(
        Damo Embedding System
        --------------------
        A high-performance embedding storage and optimization system with:
        - Efficient key-value lookups
        - Gradient application optimizations
        - Checkpointing and serialization
        - RocksDB backend support
    )pbdoc";

  py::class_<PyDamo>(m, "Damo")
      .def(py::init<const std::string&>(), py::arg("config_file"),
           R"pbdoc(
            Initialize Damo embedding system from configuration file
            
            Args:
                config_file (str): Path to JSON configuration file
                
            Raises:
                ValueError: If config file is invalid or missing required fields
                FileNotFoundError: If config file doesn't exist
                
            Example config:
                {
                    "embeddings": [
                        {
                            "group": 0,
                            "dim": 128,
                            "initializer": {"name": "xavier"},
                            "optimizer": {"name": "adamw", ...}
                        }
                    ],
                    "dir": "./data/rocksdb",
                    "ttl": 864000
                }
            )pbdoc")

      .def("pull", &PyDamo::pull, py::arg("group"), py::arg("keys"),
           py::arg("weights"),
           R"pbdoc(
            Retrieve embedding weights for given keys
            
            Args:
                group (int): Embedding group index (0-based)
                keys (np.ndarray[int64]): 1D array of embedding keys
                weights (np.ndarray[float32]): Output array for weights (must be pre-allocated)
                
            Raises:
                ValueError: For invalid group, dimension mismatch, or readonly weights array
                RuntimeError: If embedding warehouse not initialized
                
            The weights array must have size: len(keys) * embedding_dimension
            )pbdoc")

      .def("push", &PyDamo::push, py::arg("group"), py::arg("keys"),
           py::arg("gradients"),
           R"pbdoc(
            Apply gradients to specified embeddings
            
            Args:
                group (int): Embedding group index (0-based)
                keys (np.ndarray[int64]): 1D array of embedding keys
                gradients (np.ndarray[float32]): 1D array of gradient values
                
            Raises:
                ValueError: For invalid group or dimension mismatch
                RuntimeError: If gradient application fails
                
            Gradient array size must be: len(keys) * embedding_dimension
            )pbdoc")

      .def("load", &PyDamo::load, py::arg("checkpoint_path"),
           R"pbdoc(
            Load embeddings from checkpoint file (atomic operation)
            
            Args:
                checkpoint_path (str): Path to valid checkpoint file
                
            Raises:
                FileNotFoundError: If checkpoint file doesn't exist
                IOError: If file format is invalid
                RuntimeError: If database operation fails
                
            Warning:
                This will completely overwrite current embeddings
            )pbdoc")

      .def("dump", &PyDamo::dump, py::arg("file_path"),
           R"pbdoc(
            Dump embeddings to binary file format
            
            Args:
                file_path (str): Output file path
                
            Raises:
                IOError: If file write fails
                RuntimeError: If snapshot creation fails
            )pbdoc")

      .def("checkpoint", &PyDamo::checkpoint, py::arg("checkpoint_path"),
           R"pbdoc(
            Create a portable checkpoint of current state
            
            Args:
                checkpoint_path (str): Output file path
                
            Raises:
                IOError: If file creation fails
                RuntimeError: If database operation fails
                
            The checkpoint can be used with load() to restore state
            )pbdoc")
      .def(
          "__repr__",
          [](PyDamo& p) -> std::string {
            try {
              return p.to_json();
            } catch (const std::exception& e) {
              return "<Damo: Error generating representation>";
            }
          },
          R"pbdoc(
        Return JSON representation of configuration
        
        Returns:
            str: JSON string describing current configuration
        )pbdoc");

  // Register exception translations
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const std::invalid_argument& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::runtime_error& e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const std::ios_base::failure& e) {
      PyErr_SetString(PyExc_IOError, e.what());
    } catch (const std::exception& e) {
      PyErr_SetString(PyExc_Exception, e.what());
    }
  });

  // Add version info
  m.attr("__version__") = "1.2.0";
  m.def("get_version", []() { return "1.2.0"; }, "Get library version");
}
