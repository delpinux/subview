#include <Kokkos_Core.hpp>

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>

template <typename T>
class SharedArray
{
 private:
  size_t m_size;
  std::shared_ptr<T[]> m_values;

 public:
  class SharedArrayView
  {
   private:
    SharedArray<T> m_array;
    size_t m_begin;
    size_t m_size;

   public:
    inline size_t
    size() const
    {
      return m_size;
    }

    inline T&
    operator()(size_t i)
    {
      return m_array(m_begin + i);
    }

    SharedArrayView(const SharedArray& array, size_t begin, size_t size) : m_array{array}, m_begin{begin}, m_size{size}
    {}
    ~SharedArrayView() = default;
  };

  inline size_t
  size() const
  {
    return m_size;
  }

  inline T&
  operator()(size_t i)
  {
    return m_values[i];
  }

  inline friend SharedArrayView
  subView(SharedArray& array, size_t begin, size_t end)
  {
    return SharedArrayView(array, begin, end - begin);
  }

  SharedArray(const SharedArray&) = default;
  SharedArray(SharedArray&&)      = default;

  SharedArray(const size_t size) : m_size{size}, m_values{new T[size]} {}
  ~SharedArray() = default;
};

template <typename T>
class RawView
{
 private:
  T* const m_values;
  size_t m_size;

 public:
  size_t
  size() const
  {
    return m_size;
  }

  T&
  operator()(size_t i) const
  {
    return m_values[i];
  }

  RawView(const Kokkos::View<T*>& view, size_t begin, size_t end) : m_values{&(view(begin))}, m_size(end - begin) {}
  RawView(SharedArray<T>& shared_array, size_t begin, size_t end)
    : m_values{&(shared_array(begin))}, m_size(end - begin)
  {}

  ~RawView() = default;
};

int
main(int argc, char* argv[])
{
  using ValueType = size_t;

  std::map<double, std::string> time_method_map;

  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " [nb-loops]\n";
    return 1;
  }

  const size_t nb_loops = atoi(argv[1]);
  std::cout << "processing " << nb_loops << " loops\n";

  Kokkos::initialize(argc, argv);

  {
    const size_t nb_rows = 100000;
    Kokkos::View<size_t*> row_map("row_map", nb_rows + 1);
    row_map(0) = 0;
    for (size_t i = 1; i < row_map.extent(0); ++i) {
      row_map(i) = row_map(i - 1) + 5;
    }

    Kokkos::View<ValueType*> entries("entries", row_map(nb_rows));
    for (size_t i = 0; i < nb_rows; ++i) {
      for (size_t j = row_map(i); j < row_map(i + 1); ++j) {
        entries(j) = i;
      }
    }

    Kokkos::View<ValueType*> sums("sums", nb_rows);
    for (size_t i = 0; i < sums.extent(0); ++i) {
      sums[i] = 0;
    }
    Kokkos::Timer timer;

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        ValueType sum = 0;
        for (size_t k = row_map(i); k < row_map(i + 1); ++k) {
          sum += entries(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "Kokkos/Direct";

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        RawView view(entries, row_map(i), row_map(i + 1));
        ValueType sum = 0;
        for (size_t k = 0; k < view.size(); ++k) {
          sum += entries(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "Kokkos|RawView";

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        auto sub_view = Kokkos::subview(entries, std::make_pair(row_map(i), row_map(i + 1)));
        ValueType sum = 0;
        for (size_t k = 0; k < sub_view.extent(0); ++k) {
          sum += sub_view(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "Kokkos::subView";

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        Kokkos::View<ValueType*> view(entries, std::make_pair(row_map(i), row_map(i + 1)));
        ValueType sum = 0;
        for (size_t k = 0; k < view.extent(0); ++k) {
          sum += view(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "Kokkos::View";

    SharedArray<ValueType> shared_array{entries.extent(0)};
    for (size_t i = 0; i < shared_array.size(); ++i) {
      shared_array(i) = entries(i);
    }

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        auto&& view   = subView(shared_array, row_map(i), row_map(i + 1));
        ValueType sum = 0;
        for (size_t k = 0; k < view.size(); ++k) {
          sum += view(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "SharedArrayView";

    timer.reset();
    for (size_t n = 0; n < nb_loops; ++n) {
      for (size_t i = 0; i < nb_rows; ++i) {
        RawView view(shared_array, row_map(i), row_map(i + 1));
        ValueType sum = 0;
        for (size_t k = 0; k < view.size(); ++k) {
          sum += view(k);
        }
        sums(i) = sum;
      }
    }
    time_method_map[timer.seconds()] = "SharedArray|RawView";

    std::cout.precision(15);
    for (const auto& [time, name] : time_method_map) {
      std::cout << std::setw(25) << std::setfill('.') << std::left << name << ' ' << time << '\n';
    }
  }

  Kokkos::finalize();
  return 0;
}
