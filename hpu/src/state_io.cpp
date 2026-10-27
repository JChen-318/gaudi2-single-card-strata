#include "strata/hpu/state_io.hpp"
#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::hpu {
static_assert(std::endian::native==std::endian::little,"state format requires a little-endian host");
namespace {
constexpr std::array<uint8_t,8> magic{'S','T','R','H','P','U',2,0};
std::atomic<uint64_t> temporary_id{0};
constexpr auto crc_table() {
    std::array<uint64_t,256> table{};
    for(uint64_t byte=0;byte<256;++byte) {
        auto value=byte<<56;
        for(int bit=0;bit<8;++bit) value=value&0x8000000000000000ull?(value<<1)^0x42f0e1eba9ea3693ull:value<<1;
        table[byte]=value;
    }
    return table;
}
constexpr auto table=crc_table();
uint64_t crc(uint64_t value,std::span<const uint8_t> bytes) {
    for(auto byte:bytes) value=table[((value>>56)^byte)&255]^(value<<8);
    return value;
}
void write_all(int fd,std::span<const uint8_t> bytes) {
    while(!bytes.empty()) {
        const auto count=::write(fd,bytes.data(),std::min<size_t>(bytes.size(),1u<<20));
        if(count<0 && errno==EINTR) continue;
        if(count<=0) throw StateIoError("state file write failed",false,errno==ENOSPC || errno==EDQUOT);
        bytes=bytes.subspan(size_t(count));
    }
}
void read_all(int fd,std::span<uint8_t> bytes) {
    while(!bytes.empty()) {
        const auto count=::read(fd,bytes.data(),std::min<size_t>(bytes.size(),1u<<20));
        if(count<0 && errno==EINTR) continue;
        if(count<=0) throw std::runtime_error("truncated/unreadable state file");
        bytes=bytes.subspan(size_t(count));
    }
}
}
StateWriter::StateWriter(std::string path,uint64_t limit):path_(std::move(path)),limit_(limit) {
    if(path_.empty() || limit<magic.size() || limit>UINT64_MAX-8) throw std::invalid_argument("invalid state path/byte cap");
    temporary_=path_+".tmp."+std::to_string(::getpid())+"."+std::to_string(temporary_id++);
    fd_=::open(temporary_.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd_<0) throw StateIoError("cannot create state temporary file",false,errno==ENOSPC || errno==EDQUOT);
    try {bytes(magic);} catch(...) {::close(fd_);fd_=-1;::unlink(temporary_.c_str());throw;}
}
StateWriter::~StateWriter() {if(fd_>=0) ::close(fd_);if(!published_ && !temporary_.empty()) ::unlink(temporary_.c_str());}
void StateWriter::bytes(std::span<const uint8_t> values) {
    if(fd_<0 || published_ || values.size()>limit_-written_) throw std::invalid_argument("state writer closed or payload cap exceeded");
    write_all(fd_,values);written_+=values.size();checksum_=crc(checksum_,values);
}
void StateWriter::integer(uint64_t value) {bytes({reinterpret_cast<const uint8_t*>(&value),8});}
void StateWriter::text(const std::string& value) {integer(value.size());bytes({reinterpret_cast<const uint8_t*>(value.data()),value.size()});}
void StateWriter::finish() {
    if(fd_<0 || published_) throw std::logic_error("state writer already finished");
    write_all(fd_,{reinterpret_cast<const uint8_t*>(&checksum_),8});
    if(::fsync(fd_)!=0) throw StateIoError("state file fsync failed",false,errno==ENOSPC || errno==EDQUOT);
    const auto result=::close(fd_);fd_=-1;
    if(result!=0) throw StateIoError("state file close failed");
    if(::rename(temporary_.c_str(),path_.c_str())!=0) throw StateIoError("atomic state publication failed",false,errno==ENOSPC || errno==EDQUOT);
    published_=true;
    const auto parent=std::filesystem::path(path_).parent_path();
    const int directory=::open(parent.empty()?".":parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    if(directory<0) throw StateIoError("state published but parent directory cannot be opened for fsync",true);
    const auto synced=::fsync(directory);::close(directory);
    if(synced!=0) throw StateIoError("state published but parent directory fsync failed",true);
}
StateReader::StateReader(const std::string& path,uint64_t limit) {
    fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
    if(fd_<0) throw StateIoError("cannot open state file");
    try {
        struct stat status{};
        if(::fstat(fd_,&status)!=0 || !S_ISREG(status.st_mode) || status.st_size<16 || uint64_t(status.st_size)-8>limit)
            throw std::invalid_argument("invalid state file size/type or payload cap exceeded");
        remaining_=uint64_t(status.st_size)-8;std::array<uint8_t,8> header{};bytes(header);
        if(header!=magic) throw std::invalid_argument("unsupported state format/version");
    } catch(...) {::close(fd_);fd_=-1;throw;}
}
StateReader::~StateReader() {if(fd_>=0) ::close(fd_);}
void StateReader::bytes(std::span<uint8_t> values) {
    if(finished_ || values.size()>remaining_) throw std::invalid_argument("truncated state payload or reader already finished");
    read_all(fd_,values);remaining_-=values.size();checksum_=crc(checksum_,values);
}
uint64_t StateReader::integer() {uint64_t value=0;bytes({reinterpret_cast<uint8_t*>(&value),8});return value;}
std::string StateReader::text(uint64_t limit) {
    const auto size=integer();if(size>limit || size>remaining_) throw std::invalid_argument("state text length exceeds bound");
    std::string value(size,'\0');bytes({reinterpret_cast<uint8_t*>(value.data()),value.size()});return value;
}
void StateReader::finish() {
    if(finished_ || remaining_) throw std::invalid_argument("state contains unread/trailing payload");
    uint64_t expected=0;read_all(fd_,{reinterpret_cast<uint8_t*>(&expected),8});
    if(expected!=checksum_) throw std::invalid_argument("state checksum mismatch");
    // A file modified after fstat cannot append ignored data after the checksum.
    uint8_t extra=0;ssize_t count;
    do {count=::read(fd_,&extra,1);} while(count<0 && errno==EINTR);
    if(count!=0) throw std::invalid_argument("state file changed or has trailing bytes");
    finished_=true;
}
}
