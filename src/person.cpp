#include "person.h"

#include <utility>

Person::Person(std::string  name, int age) : name(std::move(name)), age(age) {}

std::string Person::introduce() const {
    return "My name is " + name + " and I am " + std::to_string(age) + " years old.";
}
