#pragma once
#include <tuple>
namespace tla {
template<size_t I,size_t J,class T> auto get(const T& x){return std::get<J>(std::get<I>(x));}
}
