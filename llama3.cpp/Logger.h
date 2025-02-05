#pragma once

#include <ostream>

class SinkStream : public std::ostream
{
public:
    SinkStream();
    //   NullStream(const NullStream &) : std::ostream(nullptr) {}
};

class Logger
{
public:
    enum Level
    {
        FATAL,
        ERROR,
        WARN,
        INFO,
        DEBUG,
        TRACE,
    };

public:
    Logger();
    Logger(Level level);

    void setLevel(Level l);
    std::ostream &operator()(Level l);

private:
    Level level;
    SinkStream sink;
};

extern Logger logger;
