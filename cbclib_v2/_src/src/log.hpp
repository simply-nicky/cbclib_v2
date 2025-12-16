#ifndef LOG_
#define LOG_
#include "include.hpp"

#ifndef CPP_LOG
#define CPP_LOG 0
#endif

namespace cbclib {

template <typename T, typename = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream &>() << std::declval<T &>())>>
    : std::true_type {};

template <typename T>
constexpr bool is_streamable_v = is_streamable<T>::value;

enum LogLevel {ERROR, WARNING, INFO, DEBUG};

class STDErrLogStream
{
private:
    static inline std::mutex s_mutex;

public:
    STDErrLogStream() = default;

    static const std::ostream & stream()
    {
        return std::cerr;
    }

    static void write(const std::string & msg)
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        std::cerr << msg;
    }
};

class FileLogStream
{
private:
    static inline std::mutex s_mutex;

public:
    static std::ofstream & stream()
    {
        static std::ofstream stream;
        return stream;
    }

    static void open(const char * filename)
    {
        stream().open(filename);
        if (!stream())
        {
            throw std::runtime_error(std::string("Failed to open log file: ") + filename);
        }
    }

    static void close()
    {
        if (stream().is_open())
        {
            stream().close();
        }
    }

    static void write(const std::string & msg)
    {
        if (!stream())
        {
            throw std::runtime_error("FileLogStream not initialized. Create LogFile to initialize.");
        }
        std::lock_guard<std::mutex> lock(s_mutex);
        stream() << msg;
    }

protected:
    std::ofstream m_stream;
};

class LogFile
{
public:
    explicit LogFile(const char * filename)
    {
        FileLogStream::open(filename);
    }

    ~LogFile()
    {
        FileLogStream::close();
    }

    LogFile(const LogFile &) = delete;
    LogFile & operator=(const LogFile &) = delete;
};

template <typename Stream, typename = decltype(std::declval<Stream &>().write(std::declval<std::string &>()))>
class Log
{
protected:
    static constexpr std::array<std::string_view, 4> level_strings {"ERROR", "WARNING", "INFO", "DEBUG"};
    static inline const std::unordered_map<std::string, LogLevel> level_map {
        {"ERROR", ERROR},
        {"WARNING", WARNING},
        {"INFO", INFO},
        {"DEBUG", DEBUG}
    };

public:
    Log() = default;

    virtual ~Log()
    {
        m_oss << std::endl;
        Stream::write(m_oss.str());
    }

    Log(const Log &) = delete;
    void operator=(const Log &) = delete;

    Log & get(LogLevel level = INFO)
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        m_oss << "time: ";
        m_oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        m_oss << ", " << to_string(level) << ": ";
        m_oss << std::string(level > DEBUG ? level - DEBUG : 0, '\t');
        return *this;
    }

    template <typename T, typename = std::enable_if_t<is_streamable_v<T>>>
    friend Log & operator<<(Log & log, const T & msg)
    {
        log.m_oss << msg;
        return log;
    }

    template <typename Container, typename T = typename Container::value_type, typename = std::enable_if_t<
        !is_streamable_v<Container> && is_streamable_v<T>>
    >
    friend Log & operator<<(Log & log, const Container & vec)
    {
        log.m_oss << "{";
        bool first = true;
        for (const auto & item : vec)
        {
            if (!first) log.m_oss << ", ";
            log.m_oss << item;
            first = false;
        }
        log.m_oss << "}";
        return log;
    }

public:
    static LogLevel & reporting_level()
    {
        static LogLevel m_level = DEBUG;
        return m_level;
    }

    static std::string to_string(LogLevel level)
    {
        return std::string(level_strings[level]);
    }

    static LogLevel from_string(const std::string & level)
    {
        auto it = level_map.find(level);
        if (it != level_map.end()) return it->second;

        Log<Stream>().get(WARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
        return INFO;
    }

protected:
    std::ostringstream m_oss;
};

}

#define LOG_INIT(level) \
    cbclib::Log<cbclib::STDErrLogStream>::reporting_level() = level

#define LOG(level) \
    if constexpr (CPP_LOG == 0) ; \
    else if (level < cbclib::Log<cbclib::STDErrLogStream>::reporting_level()) ; \
    else cbclib::Log<cbclib::STDErrLogStream>().get(level)

#define LOG_FILE(level) \
    if constexpr (CPP_LOG == 0) ; \
    else if (level < cbclib::Log<cbclib::FileLogStream>::reporting_level()) ; \
    else cbclib::Log<cbclib::FileLogStream>().get(level)

#endif //LOG_
