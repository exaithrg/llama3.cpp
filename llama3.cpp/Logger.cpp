#include <iostream>

#include "Logger.h"

Logger logger;

SinkStream::SinkStream()
    : std::ostream(nullptr)
{
}

Logger::Logger()
    : Logger(ERROR)
{
}

Logger::Logger(Level level)
    : level(level)
{
}

void Logger::setLevel(Level l) { level = l; }

std::ostream &Logger::operator()(Level l)
{
    if (l <= level)
        return std::cout;

    return sink;
}
