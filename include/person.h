#ifndef PERSON_H
#define PERSON_H

#ifdef _WIN32
#ifdef DLL_EXPORT
    #define API __declspec(dllexport)
  #else
    #define API __declspec(dllimport)
  #endif
#else
#define API
#endif

#include <string>

class API Person {
public:
    Person(std::string  name, int age);
    std::string introduce() const;

private:
    std::string name;
    int age;
};

#endif // PERSON_H
