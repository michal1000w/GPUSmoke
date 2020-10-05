//////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2018-2020 Syoyo Fujita
// Copyright (c) 2012-2018 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Syoyo Fujita and DreamWorks Animation nor the names
// of its contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
#ifndef TINY_VDB_IO_H_
#define TINY_VDB_IO_H_

#include <bitset>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER) || defined(__MINGW32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

namespace tinyvdb {

    // For voxel coordinate.
    struct Vec3i {
        int x;
        int y;
        int z;
    };

    class Boundsi {
    public:
        Boundsi() {
            bmin.x = std::numeric_limits<int>::max();
            bmin.y = std::numeric_limits<int>::max();
            bmin.z = std::numeric_limits<int>::max();

            bmax.x = -std::numeric_limits<int>::max();
            bmax.y = -std::numeric_limits<int>::max();
            bmax.z = -std::numeric_limits<int>::max();
        }

        ///
        /// Returns true if given coordinate is within this bound
        ///
        bool Contains(const Vec3i& v) {
            if ((bmin.x <= v.x) && (v.x <= bmax.x) && (bmin.y <= v.y) &&
                (v.y <= bmax.y) && (bmin.z <= v.z) && (v.z <= bmax.z)) {
                return true;
            }

            return false;
        }

        ///
        /// Returns true if given bounding box overlaps with this bound.
        ///
        bool Overlaps(const Boundsi& b) {
            if (Contains(b.bmin) || Contains(b.bmax)) {
                return true;
            }
            return false;
        }

        ///
        /// Compute union of two bounds.
        ///
        static Boundsi Union(const Boundsi& a, const Boundsi& b) {
            Boundsi bound;

            bound.bmin.x = std::min(a.bmin.x, b.bmin.x);
            bound.bmin.y = std::min(a.bmin.y, b.bmin.y);
            bound.bmin.z = std::min(a.bmin.z, b.bmin.z);

            bound.bmax.x = std::max(a.bmax.x, b.bmax.x);
            bound.bmax.y = std::max(a.bmax.y, b.bmax.y);
            bound.bmax.z = std::max(a.bmax.z, b.bmax.z);

            return bound;
        }

        friend std::ostream& operator<<(std::ostream& os, const Boundsi& bound);

        Vec3i bmin;
        Vec3i bmax;
    };

    // --- vvv --------------------------------------------------

    /*
    The MIT License (MIT)

    Copyright (c) 2019 Syoyo Fujita.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    */

    ///
    /// @brief dynamically allocatable bitset
    ///
    class dynamic_bitset {
    public:
        dynamic_bitset() = default;
        dynamic_bitset(dynamic_bitset&&) = default;
        dynamic_bitset(const dynamic_bitset&) = default;

        dynamic_bitset& operator=(const dynamic_bitset&) = default;

        ~dynamic_bitset() = default;

        ///
        /// @brief Construct dynamic_bitset with given number of bits.
        ///
        /// @param[in] nbits The number of bits to use.
        /// @param[in] value Initize bitfield with this value.
        ///
        explicit dynamic_bitset(size_t nbits, uint64_t value) {
            _num_bits = nbits;

            size_t num_bytes;
            if (nbits < 8) {
                num_bytes = 1;
            }
            else {
                num_bytes = 1 + (nbits - 1) / 8;
            }

            _data.resize(num_bytes);

            // init with zeros
            std::fill_n(_data.begin(), _data.size(), 0);

            // init with `value`.

            if (nbits < sizeof(uint64_t)) {
                assert(num_bytes < 3);

                uint64_t masked_value = value & ((1 << (nbits + 1)) - 1);

                for (size_t i = 0; i < _data.size(); i++) {
                    _data[i] = (masked_value >> (i * 8)) & 0xff;
                }

            }
            else {
                for (size_t i = 0; i < sizeof(uint64_t); i++) {
                    _data[i] = (value >> (i * 8)) & 0xff;
                }
            }
        }

        ///
        /// Equivalent to std::bitset::any()
        ///
        bool any() const {
            for (size_t i = 0; i < _num_bits; i++) {
                if ((*this)[i]) {
                    return true;
                }
            }

            return false;
        }

        ///
        /// Equivalent to std::bitset::all()
        ///
        bool all() const {
            for (size_t i = 0; i < _num_bits; i++) {
                if (false == (*this)[i]) {
                    return false;
                }
            }

            return true;
        }

        ///
        /// Equivalent to std::bitset::none()
        ///
        bool none() const {
            for (size_t i = 0; i < _num_bits; i++) {
                if ((*this)[i]) {
                    return false;
                }
            }

            return true;
        }

        ///
        /// Equivalent to std::bitset::flip()
        ///
        dynamic_bitset& flip() {
            for (size_t i = 0; i < _num_bits; i++) {
                set(i, (*this)[i] ? false : true);
            }

            return (*this);
        }

        ///
        /// @brief Resize dynamic_bitset.
        ///
        /// @details Resize dynamic_bitset. Resize behavior is similar to
        /// std::vector::resize.
        ///
        /// @param[in] nbits The number of bits to use.
        ///
        void resize(size_t nbits) {
            _num_bits = nbits;

            size_t num_bytes;
            if (nbits < 8) {
                num_bytes = 1;
            }
            else {
                num_bytes = 1 + (nbits - 1) / 8;
            }

            _data.resize(num_bytes);
        }

        ///
        /// @return The number of bits that are set to `true`
        ///
        uint32_t count() const {
            uint32_t c = 0;

            for (size_t i = 0; i < _num_bits; i++) {
                c += (*this)[i] ? 1 : 0;
            }

            return c;
        }

        bool test(size_t pos) const {
            // TODO(syoyo): Do range check and throw when out-of-bounds access.
            return (*this)[pos];
        }

        void reset() { std::fill_n(_data.begin(), _data.size(), 0); }

        // Set all bitfield with `value`
        void setall(bool value) {
            for (size_t i = 0; i < _num_bits; i++) {
                set(i, value);
            }
        }

        void set(size_t pos, bool value = true) {
            size_t byte_loc = pos / 8;
            uint8_t offset = pos % 8;

            uint8_t bitfield = uint8_t(1 << offset);

            if (value == true) {
                // bit on
                _data[byte_loc] |= bitfield;
            }
            else {
                // turn off bit
                _data[byte_loc] &= (~bitfield);
            }
        }

        std::string to_string() const {
            std::stringstream ss;

            for (size_t i = 0; i < _num_bits; i++) {
                ss << ((*this)[_num_bits - i - 1] ? "1" : "0");
            }

            return ss.str();
        }

        bool operator[](size_t pos) const {
            size_t byte_loc = pos / 8;
            size_t offset = pos % 8;

            return (_data[byte_loc] >> offset) & 0x1;
        }

        // Return the number of bits.
        size_t nbits() const { return _num_bits; }

        // Return storage size.
        size_t size() const { return _data.size(); }

        // Return memory address of bitfield(as an byte array)
        const uint8_t* data() const { return _data.data(); }

        // Return memory address of bitfield(as an byte array)
        uint8_t* data() { return _data.data(); }

    private:
        size_t _num_bits{ 0 };

        // bitfields are reprentated as an array of bytes.
        std::vector<uint8_t> _data;
    };

    // --^^^---------------------------------------------------------------

    // TODO(syoyo): Move to IMPLEMENTATION
#define TINYVDBIO_ASSERT(x) assert(x)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wc++11-long-long"
#endif

    typedef struct {
        uint32_t file_version;
        uint32_t major_version;
        uint32_t minor_version;
        // bool has_grid_offsets;
        bool is_compressed;
        bool half_precision;
        std::string uuid;
        uint64_t offset_to_data;  // Byte offset to VDB data
    } VDBHeader;

    typedef struct {
    } VDBMeta;

    typedef enum {
        TINYVDBIO_SUCCESS,
        TINYVDBIO_ERROR_INVALID_FILE,
        TINYVDBIO_ERROR_INVALID_HEADER,
        TINYVDBIO_ERROR_INVALID_DATA,
        TINYVDBIO_ERROR_INVALID_ARGUMENT,
        TINYVDBIO_ERROR_UNIMPLEMENTED
    } VDBStatus;

    // forward decl.
    class StreamReader;
    class StreamWriter;
    struct DeserializeParams;

    ////////////////////////////////////////

    /// internal Per-node indicator byte that specifies what additional metadata
    /// is stored to permit reconstruction of inactive values
    enum {
        /*0*/ NO_MASK_OR_INACTIVE_VALS,  // no inactive vals, or all inactive vals are
                                         // +background
                                         /*1*/ NO_MASK_AND_MINUS_BG,      // all inactive vals are -background
                                         /*2*/ NO_MASK_AND_ONE_INACTIVE_VAL,  // all inactive vals have the same
                                                                              // non-background val
                                                                              /*3*/ MASK_AND_NO_INACTIVE_VALS,     // mask selects between -background and
                                                                                                                   // +background
                                                                                                                   /*4*/ MASK_AND_ONE_INACTIVE_VAL,  // mask selects between backgd and one other
                                                                                                                                                     // inactive val
                                                                                                                                                     /*5*/ MASK_AND_TWO_INACTIVE_VALS,  // mask selects between two non-background
                                                                                                                                                                                        // inactive vals
                                                                                                                                                                                        /*6*/ NO_MASK_AND_ALL_VALS  // > 2 inactive vals, so no mask compression at
                                                                                                                                                                                                                    // all
    };

    // TODO(syoyo): remove

    /// @brief Bit mask for the internal and leaf nodes of VDB. This
    /// is a 64-bit implementation.
    ///
    /// @note A template specialization for Log2Dim=1 and Log2Dim=2 are
    /// given below.
    // template <int Log2Dim>
    class NodeMask {
    public:
        // static_assert(Log2Dim > 2, "expected NodeMask template specialization, got
        // base template");
        int32_t LOG2DIM;
        int32_t DIM;
        int32_t BITSIZE;
        // int32_t WORD_COUNT;

        // static const int32_t LOG2DIM = Log2Dim;
        // static const int32_t DIM = 1 << Log2Dim;
        // static const int32_t SIZE = 1 << 3 * Log2Dim;
        // static const int32_t WORD_COUNT = SIZE >> 6;  // 2^6=64
        // using Word = int64;
        // typedef int64 Word;

    private:
        // The bits are represented as a linear array of Words, and the
        // size of a Word is 32 or 64 bits depending on the platform.
        // The BIT_MASK is defined as the number of bits in a Word - 1
        // static const int32_t BIT_MASK   = sizeof(void*) == 8 ? 63 : 31;
        // static const int32_t LOG2WORD   = BIT_MASK == 63 ? 6 : 5;
        // static const int32_t WORD_COUNT = SIZE >> LOG2WORD;
        // using Word = boost::mpl::if_c<BIT_MASK == 63, int64, int32>::type;

        // std::vector<Word> mWords;  // only member data!
        // std::vector<bool> bits;
        dynamic_bitset bits;

    public:
        NodeMask() {
            LOG2DIM = 0;
            DIM = 0;
            BITSIZE = 0;
            // WORD_COUNT = 0;
        }

        void Alloc(int32_t log2dim) {
            LOG2DIM = log2dim;
            DIM = 1 << log2dim;
            BITSIZE = 1 << 3 * log2dim;
            // WORD_COUNT = SIZE >> 6;  // 2^6=64

            // mWords.resize(WORD_COUNT);

            bits.resize(size_t(BITSIZE));
            bits.reset();
        }

        /// Default constructor sets all bits off
        NodeMask(int32_t log2dim) {
            LOG2DIM = log2dim;
            DIM = 1 << log2dim;
            BITSIZE = 1 << 3 * log2dim;
            // WORD_COUNT = SIZE >> 6;  // 2^6=64

            bits.resize(size_t(BITSIZE));
            bits.reset();
        }

        /// All bits are set to the specified state
        NodeMask(int32_t log2dim, bool on) {
            LOG2DIM = log2dim;
            DIM = 1 << log2dim;
            BITSIZE = 1 << 3 * log2dim;
            // WORD_COUNT = SIZE >> 6;  // 2^6=64

            bits.resize(size_t(BITSIZE));
            bits.setall(on);
        }
        /// Copy constructor
        NodeMask(const NodeMask& other) { *this = other; }
        /// Destructor
        ~NodeMask() {}
        /// Assignment operator
        NodeMask& operator=(const NodeMask& other) {
            LOG2DIM = other.LOG2DIM;
            DIM = other.DIM;
            BITSIZE = other.BITSIZE;
            // WORD_COUNT = other.WORD_COUNT;

            bits = other.bits;
            // mWords = other.mWords;
            return *this;
            // int32_t n = WORD_COUNT;
            // const Word* w2 = other.mWords;
            // for (Word *w1 = mWords; n--; ++w1, ++w2) *w1 = *w2;
        }

#if 0
        using OnIterator = OnMaskIterator<NodeMask>;
        using OffIterator = OffMaskIterator<NodeMask>;
        using DenseIterator = DenseMaskIterator<NodeMask>;

        OnIterator beginOn() const { return OnIterator(this->findFirstOn(), this); }
        OnIterator endOn() const { return OnIterator(SIZE, this); }
        OffIterator beginOff() const { return OffIterator(this->findFirstOff(), this); }
        OffIterator endOff() const { return OffIterator(SIZE, this); }
        DenseIterator beginDense() const { return DenseIterator(0, this); }
        DenseIterator endDense() const { return DenseIterator(SIZE, this); }

        bool operator==(const NodeMask& other) const {
            int n = int(WORD_COUNT);
            for (const Word* w1 = mWords.data(), *w2 = other.mWords.data();
                n-- && *w1++ == *w2++;)
                ;
            return n == -1;
        }

        bool operator!=(const NodeMask& other) const { return !(*this == other); }
#endif

#if 0  // remove
        //
        // Bitwise logical operations
        //

        /// @brief Apply a functor to the words of the this and the other mask.
        ///
        /// @details An example that implements the "operator&=" method:
        /// @code
        /// struct Op { inline void operator()(W &w1, const W& w2) const { w1 &= w2; }
        /// };
        /// @endcode
        template <typename WordOp>
        const NodeMask& foreach(const NodeMask& other, const WordOp& op) {
            Word* w1 = mWords.data();
            const Word* w2 = other.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2) op(*w1, *w2);
            return *this;
        }
        template <typename WordOp>
        const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2,
            const WordOp& op) {
            Word* w1 = mWords.data();
            const Word* w2 = other1.mWords.data(), * w3 = other2.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2, ++w3) op(*w1, *w2, *w3);
            return *this;
        }
        template <typename WordOp>
        const NodeMask& foreach(const NodeMask& other1, const NodeMask& other2,
            const NodeMask& other3, const WordOp& op) {
            Word* w1 = mWords.data();
            const Word* w2 = other1.mWords.data(), * w3 = other2.mWords.data(),
                * w4 = other3.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2, ++w3, ++w4)
                op(*w1, *w2, *w3, *w4);
            return *this;
        }
        /// @brief Bitwise intersection
        const NodeMask& operator&=(const NodeMask& other) {
            Word* w1 = mWords.data();
            const Word* w2 = other.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 &= *w2;
            return *this;
        }
        /// @brief Bitwise union
        const NodeMask& operator|=(const NodeMask& other) {
            Word* w1 = mWords.data();
            const Word* w2 = other.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 |= *w2;
            return *this;
        }
        /// @brief Bitwise difference
        const NodeMask& operator-=(const NodeMask& other) {
            Word* w1 = mWords.data();
            const Word* w2 = other.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 &= ~*w2;
            return *this;
        }
        /// @brief Bitwise XOR
        const NodeMask& operator^=(const NodeMask& other) {
            Word* w1 = mWords.data();
            const Word* w2 = other.mWords.data();
            for (int32_t n = WORD_COUNT; n--; ++w1, ++w2) *w1 ^= *w2;
            return *this;
        }
        NodeMask operator!() const {
            NodeMask m(*this);
            m.toggle();
            return m;
        }
        NodeMask operator&(const NodeMask& other) const {
            NodeMask m(*this);

            m &= other;
            return m;
        }
        NodeMask operator|(const NodeMask& other) const {
            NodeMask m(*this);
            m |= other;
            return m;
        }
        NodeMask operator^(const NodeMask& other) const {
            NodeMask m(*this);
            m ^= other;
            return m;
        }
#endif

        /// Return the byte size of this NodeMask
        size_t memUsage() const { return bits.size(); }

#if 0
        /// Return the total number of on bits
        int32_t countOn() const {
            int32_t sum = 0, n = WORD_COUNT;
            std::cout << "cnt = " << n << ", sz = " << mWords.size() << std::endl;
            for (const Word* w = mWords.data(); n--; ++w) sum += CountOn(*w);
            return sum;
        }

        /// Return the total number of on bits
        int32_t countOff() const { return SIZE - this->countOn(); }
        /// Set the <i>n</i>th  bit on
        void setOn(int32_t n) {
            TINYVDBIO_ASSERT((n >> 6) < WORD_COUNT);
            mWords[n >> 6] |= Word(1) << (n & 63);
        }
        /// Set the <i>n</i>th bit off
        void setOff(int32_t n) {
            TINYVDBIO_ASSERT((n >> 6) < WORD_COUNT);
            mWords[n >> 6] &= ~(Word(1) << (n & 63));
        }
        /// Set the <i>n</i>th bit to the specified state
        void set(int32_t n, bool On) { On ? this->setOn(n) : this->setOff(n); }
        /// Set all bits to the specified state
        void set(bool on) {
            const Word state = on ? ~Word(0) : Word(0);
            int32_t n = WORD_COUNT;
            for (Word* w = mWords.data(); n--; ++w) *w = state;
        }
        /// Set all bits on
        void setOn() {
            int32_t n = WORD_COUNT;
            for (Word* w = mWords.data(); n--; ++w) *w = ~Word(0);
        }
        /// Set all bits off
        void setOff() {
            int32_t n = WORD_COUNT;
            for (Word* w = mWords.data(); n--; ++w) *w = Word(0);
        }
        /// Toggle the state of the <i>n</i>th bit
        void toggle(int32_t n) {
            TINYVDBIO_ASSERT((n >> 6) < WORD_COUNT);
            mWords[n >> 6] ^= Word(1) << (n & 63);
        }
        /// Toggle the state of all bits in the mask
        void toggle() {
            int32_t n = WORD_COUNT;
            for (Word* w = mWords.data(); n--; ++w) *w = ~*w;
        }
        /// Set the first bit on
        void setFirstOn() { this->setOn(0); }
        /// Set the last bit on
        void setLastOn() { this->setOn(SIZE - 1); }
        /// Set the first bit off
        void setFirstOff() { this->setOff(0); }
        /// Set the last bit off
        void setLastOff() { this->setOff(SIZE - 1); }
        /// Return @c true if the <i>n</i>th bit is on
#endif

        uint32_t nbits() const { return uint32_t(bits.nbits()); }

        uint32_t count_on() const { return uint32_t(bits.count()); }

        bool is_on(int32_t n) const {
            TINYVDBIO_ASSERT(n < int32_t(bits.nbits()));
            return bits.test(size_t(n));
        }

        bool is_off(int32_t n) const { return !this->is_on(n); }

#if 0
        /// Return @c true if all the bits are on
        bool isOn() const {
            int n = int(WORD_COUNT);
            for (const Word* w = mWords.data(); n-- && *w++ == ~Word(0);)
                ;
            return n == -1;
        }
        /// Return @c true if all the bits are off
        bool isOff() const {
            int n = int(WORD_COUNT);
            for (const Word* w = mWords.data(); n-- && *w++ == Word(0);)
                ;
            return n == -1;
        }
        /// Return @c true if bits are either all off OR all on.
        /// @param isOn Takes on the values of all bits if the method
        /// returns true - else it is undefined.
        bool isConstant(bool& isOn) const {
            isOn = (mWords[0] == ~Word(0));  // first word has all bits on
            if (!isOn && mWords[0] != Word(0)) return false;  // early out
            const Word* w = mWords.data() + 1, * n = mWords.data() + WORD_COUNT;
            while (w < n && *w == mWords[0]) ++w;
            return w == n;
        }
        int32_t findFirstOn() const {
            int32_t n = 0;
            const Word* w = mWords.data();
            for (; n < WORD_COUNT && !*w; ++w, ++n)
                ;
            return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
        }
        int32_t findFirstOff() const {
            int32_t n = 0;
            const Word* w = mWords.data();
            for (; n < WORD_COUNT && !~*w; ++w, ++n)
                ;
            return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(~*w);
        }

        //@{
        /// Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
        template <typename WordT>
        WordT getWord(int n) const {
            TINYVDBIO_ASSERT(n * 8 * sizeof(WordT) < SIZE);
            return reinterpret_cast<const WordT*>(mWords)[n];
        }
        template <typename WordT>
        WordT& getWord(int n) {
            TINYVDBIO_ASSERT(n * 8 * sizeof(WordT) < SIZE);
            return reinterpret_cast<WordT*>(mWords)[n];
        }
        //@}
#endif

  // void save(std::ostream &os) const {
  //  os.write(reinterpret_cast<const char *>(bits.data()), this->size());
  //}
        bool load(StreamReader* sr);

        void seek(std::istream& is) const {
            is.seekg(int64_t(bits.size()), std::ios_base::cur);
        }

        /// @brief simple print method for debugging
        void printInfo(std::ostream& os = std::cout) const {
            os << "NodeMask: Dim=" << DIM << " Log2Dim=" << LOG2DIM
                << " Bit count=" << BITSIZE << std::endl;
        }

        std::string bitString() const {
            return bits.to_string();
        }

#if 0
        void printBits(std::ostream& os = std::cout, int32_t max_out = 80u) const {
            const int32_t n = (SIZE > max_out ? max_out : SIZE);
            for (int32_t i = 0; i < n; ++i) {
                if (!(i & 63))
                    os << "||";
                else if (!(i % 8))
                    os << "|";
                os << this->isOn(i);
            }
            os << "|" << std::endl;
        }
        void printAll(std::ostream& os = std::cout, int32_t max_out = 80u) const {
            this->printInfo(os);
            this->printBits(os, max_out);
        }

        int32_t findNextOn(int32_t start) const {
            int32_t n = start >> 6;              // initiate
            if (n >= WORD_COUNT) return SIZE;  // check for out of bounds
            int32_t m = start & 63;
            Word b = mWords[n];
            if (b & (Word(1) << m)) return start;          // simpel case: start is on
            b &= ~Word(0) << m;                            // mask out lower bits
            while (!b && ++n < WORD_COUNT) b = mWords[n];  // find next none-zero word
            return (!b ? SIZE : (n << 6) + FindLowestOn(b));  // catch last word=0
        }

        int32_t findNextOff(int32_t start) const {
            int32_t n = start >> 6;              // initiate
            if (n >= WORD_COUNT) return SIZE;  // check for out of bounds
            int32_t m = start & 63;
            Word b = ~mWords[n];
            if (b & (Word(1) << m)) return start;           // simpel case: start is on
            b &= ~Word(0) << m;                             // mask out lower bits
            while (!b && ++n < WORD_COUNT) b = ~mWords[n];  // find next none-zero word
            return (!b ? SIZE : (n << 6) + FindLowestOn(b));  // catch last word=0
        }
#endif
    };  // NodeMask

#if 0
    template <std::size_t N>
    class BitMask {
    public:
        BitMask() {}
        ~BitMask() {}

        ///
        /// Loads bit mask value from stream reader.
        ///
        bool load(StreamReader* sr);

    private:
        std::bitset<N> mask_;
    };
#endif

    class GridDescriptor {
    public:
        GridDescriptor();
        GridDescriptor(const std::string& name, const std::string& grid_type,
            bool save_float_as_half = false);
        // GridDescriptor(const GridDescriptor &rhs);
        // GridDescriptor& operator=(const GridDescriptor &rhs);
        //~GridDescriptor();

        const std::string& GridName() const { return grid_name_; }

        bool IsInstance() const { return !instance_parent_name_.empty(); }

        bool SaveFloatAsHalf() const { return save_float_as_half_; }

        uint64_t GridByteOffset() const { return grid_byte_offset_; }

        uint64_t BlockByteOffset() const { return block_byte_offset_; }

        uint64_t EndByteOffset() const { return end_byte_offset_; }

        // "\x1e" = ASCII "record separator"
        static std::string AddSuffix(const std::string& name, int n, const std::string& seperator = "\x1e");
        static std::string StripSuffix(const std::string& name, const std::string& separator = "\x1e");

        ///
        /// Read GridDescriptor from a stream.
        ///
        bool Read(StreamReader* sr, const uint32_t file_version, std::string* err);

    private:
        std::string grid_name_;
        std::string unique_name_;
        std::string instance_parent_name_;
        std::string grid_type_;

        bool save_float_as_half_;  // use fp16?
        uint64_t grid_byte_offset_;
        uint64_t block_byte_offset_;
        uint64_t end_byte_offset_;
    };

    typedef enum {
        NODE_TYPE_ROOT = 0,
        NODE_TYPE_INTERNAL = 1,
        NODE_TYPE_LEAF = 2,
        NODE_TYPE_INVALID = 3
    } NodeType;

    typedef enum {
        VALUE_TYPE_NULL = 0,
        VALUE_TYPE_FLOAT = 1,
        VALUE_TYPE_HALF = 2,
        VALUE_TYPE_BOOL = 3,
        VALUE_TYPE_DOUBLE = 4,
        VALUE_TYPE_INT = 5,
        VALUE_TYPE_STRING = 6
    } ValueType;

    static size_t GetValueTypeSize(const ValueType type) {
        if (type == VALUE_TYPE_FLOAT) {
            return sizeof(float);
        }
        else if (type == VALUE_TYPE_HALF) {
            return sizeof(short);
        }
        else if (type == VALUE_TYPE_BOOL) {
            return 1;
        }
        else if (type == VALUE_TYPE_DOUBLE) {
            return sizeof(double);
        }
        else if (type == VALUE_TYPE_STRING) {
            // string is not supported in this function.
            // Use Value::Size() instead.
            return 0;
        }
        return 0;
    }

    // Simple class to represent value object
    class Value {
    public:
        Value() : type_(VALUE_TYPE_NULL) {}

        explicit Value(bool b) : type_(VALUE_TYPE_BOOL) { boolean_value_ = b; }
        explicit Value(float f) : type_(VALUE_TYPE_FLOAT) { float_value_ = f; }
        explicit Value(double d) : type_(VALUE_TYPE_DOUBLE) { double_value_ = d; }
        explicit Value(int n) : type_(VALUE_TYPE_INT) { int_value_ = n; }
        explicit Value(const std::string& str) : type_(VALUE_TYPE_STRING) {
            string_value_ = str;
        }

        ValueType Type() const { return type_; }

        bool IsBool() const { return (type_ == VALUE_TYPE_BOOL); }
        bool IsFloat() const { return (type_ == VALUE_TYPE_FLOAT); }
        bool IsDouble() const { return (type_ == VALUE_TYPE_DOUBLE); }
        bool IsInt() const { return (type_ == VALUE_TYPE_INT); }
        bool IsString() const { return (type_ == VALUE_TYPE_STRING); }

        // Accessor
        template <typename T>
        const T& Get() const;
        template <typename T>
        T& Get();

        size_t Size() const {
            size_t len = 0;
            switch (type_) {
            case VALUE_TYPE_BOOL:
                len = 1;
                break;
            case VALUE_TYPE_HALF:
                len = sizeof(short);
                break;
            case VALUE_TYPE_INT:
                len = sizeof(int);
                break;
            case VALUE_TYPE_FLOAT:
                len = sizeof(float);
                break;
            case VALUE_TYPE_DOUBLE:
                len = sizeof(double);
                break;
            case VALUE_TYPE_STRING:
                len = string_value_.size();
                break;
            case VALUE_TYPE_NULL:
                len = 0;
                break;
            }

            return len;
        }

    protected:
        ValueType type_;

        int int_value_;
        float float_value_;
        double double_value_;
        bool boolean_value_;
        std::string string_value_;
    };

#define TINYVDB_VALUE_GET(ctype, var)             \
  template <>                                     \
  inline const ctype &Value::Get<ctype>() const { \
    return var;                                   \
  }                                               \
  template <>                                     \
  inline ctype &Value::Get<ctype>() {             \
    return var;                                   \
  }
    TINYVDB_VALUE_GET(bool, boolean_value_)
        TINYVDB_VALUE_GET(double, double_value_)
        TINYVDB_VALUE_GET(int, int_value_)
        TINYVDB_VALUE_GET(float, float_value_)
        TINYVDB_VALUE_GET(std::string, string_value_)
#undef TINYVDB_VALUE_GET

        static std::ostream& operator<<(std::ostream& os, const Value& value) {
        if (value.Type() == VALUE_TYPE_NULL) {
            os << "NULL";
        }
        else if (value.Type() == VALUE_TYPE_BOOL) {
            os << value.Get<bool>();
        }
        else if (value.Type() == VALUE_TYPE_FLOAT) {
            os << value.Get<float>();
        }
        else if (value.Type() == VALUE_TYPE_INT) {
            os << value.Get<int>();
        }
        else if (value.Type() == VALUE_TYPE_DOUBLE) {
            os << value.Get<double>();
        }

        return os;
    }

    static Value Negate(const Value& value) {
        if (value.Type() == VALUE_TYPE_NULL) {
            return value;
        }
        else if (value.Type() == VALUE_TYPE_BOOL) {
            return Value(value.Get<bool>() ? false : true);
        }
        else if (value.Type() == VALUE_TYPE_FLOAT) {
            return Value(-value.Get<float>());
        }
        else if (value.Type() == VALUE_TYPE_INT) {
            return Value(-value.Get<int>());
        }
        else if (value.Type() == VALUE_TYPE_DOUBLE) {
            return Value(-value.Get<double>());
        }

        // ???
        return value;
    }

    class TreeDesc;

    class TreeDesc {
    public:
        TreeDesc();
        ~TreeDesc();

    private:
        TreeDesc* child_tree_desc_;
    };

    class NodeInfo {
    public:
        NodeInfo(NodeType node_type, ValueType value_type, int32_t log2dim)
            : node_type_(node_type), value_type_(value_type), log2dim_(log2dim) {}

        NodeType node_type() const { return node_type_; }

        ValueType value_type() const { return value_type_; }

        int32_t log2dim() const { return log2dim_; }

    private:
        NodeType node_type_;
        ValueType value_type_;
        int32_t log2dim_;
    };

    ///
    /// Stores layout of grid hierarchy.
    ///
    class GridLayoutInfo {
    public:
        GridLayoutInfo() {}
        //~GridLayoutInfo() {}

        void Add(const NodeInfo& node_info) { node_infos_.push_back(node_info); }

        const NodeInfo& GetNodeInfo(int level) const {
            TINYVDBIO_ASSERT(level <= int(node_infos_.size()));
            return node_infos_[size_t(level)];
        }

        int NumLevels() const { return int(node_infos_.size()); }

        // Compute global voxel size for a given level.
        uint32_t ComputeGlobalVoxelSize(int level) {
            if (level >= NumLevels()) {
                // Invalid input
                return 0;
            }

            uint32_t voxel_size = 1 << node_infos_[size_t(level)].log2dim();
            for (int l = level + 1; l < NumLevels(); l++) {
                uint32_t sz = 1 << node_infos_[size_t(l)].log2dim();

                voxel_size *= sz;
            }

            return voxel_size;
        }

        std::vector<NodeInfo> node_infos_;
    };

    class InternalOrLeafNode;

    class Node {
    public:
        ///
        /// Requires GridLayoutInfo, which contains whole hierarcical grid
        /// layout information.
        ///
        Node(const GridLayoutInfo& layout_info) : grid_layout_info_(layout_info) {}
        Node& operator=(const Node& rhs) {
            grid_layout_info_ = rhs.grid_layout_info_;
            return (*this);
        }
        Node(const Node& rhs) : grid_layout_info_(rhs.grid_layout_info_) {}

        virtual ~Node() {};

        virtual bool ReadTopology(StreamReader* sr, int level,
            const DeserializeParams& params, std::string* warn,
            std::string* err) = 0;

        virtual bool ReadBuffer(StreamReader* sr, int level,
            const DeserializeParams& params, std::string* warn,
            std::string* err) = 0;

        const GridLayoutInfo& GetGridLayoutInfo() const { return grid_layout_info_; }

    protected:
        GridLayoutInfo grid_layout_info_;
    };

    //Node::~Node() {}

    ///
    /// InternalOrLeaf node represents bifurcation or leaf node.
    ///
    class InternalOrLeafNode : public Node {
    public:
        // static const int LOG2DIM = Log2Dim,  // log2 of tile count in one
        // dimension
        //    TOTAL = Log2Dim +
        //            ChildNodeType::TOTAL,  // log2 of voxel count in one dimension
        //    DIM = 1 << TOTAL,              // total voxel count in one dimension
        //    NUM_VALUES =
        //        1 << (3 * Log2Dim),  // total voxel count represented by this node
        //    LEVEL = 1 + ChildNodeType::LEVEL;  // level 0 = leaf
        // static const int64 NUM_VOXELS =
        //    uint64_t(1) << (3 * TOTAL);  // total voxel count represented by this
        //    node
        // static const int NUM_VALUES = 1 << (3 * Log2Dim); // total voxel count
        // represented by this node

        InternalOrLeafNode(const GridLayoutInfo& grid_layout_info)
            : Node(grid_layout_info) {
            origin_[0] = 0.0f;
            origin_[1] = 0.0f;
            origin_[2] = 0.0f;
            // node_values_.resize(child_mask_.size());

            num_voxels_ = 0;
        }

        InternalOrLeafNode(const InternalOrLeafNode& rhs)
            : Node(rhs.grid_layout_info_) {
            origin_[0] = rhs.origin_[0];
            origin_[1] = rhs.origin_[1];
            origin_[2] = rhs.origin_[2];

            child_nodes_ = rhs.child_nodes_;
            child_mask_ = rhs.child_mask_;

            num_voxels_ = rhs.num_voxels_;

            node_values_ = rhs.node_values_;

            value_mask_ = rhs.value_mask_;

            data_ = rhs.data_;
        }

        InternalOrLeafNode& operator=(const InternalOrLeafNode& rhs) {
            origin_[0] = rhs.origin_[0];
            origin_[1] = rhs.origin_[1];
            origin_[2] = rhs.origin_[2];

            child_nodes_ = rhs.child_nodes_;
            child_mask_ = rhs.child_mask_;

            num_voxels_ = rhs.num_voxels_;

            node_values_ = rhs.node_values_;

            value_mask_ = rhs.value_mask_;

            data_ = rhs.data_;

            return (*this);
        }

        //~InternalOrLeafNode();

        /// Deep copy function
        InternalOrLeafNode& Copy(const InternalOrLeafNode& rhs);

        ///
        /// @param[in] level Depth of this node(0: root, 1: first intermediate, ...)
        ///
        bool ReadTopology(StreamReader* sr, int level, const DeserializeParams& parms,
            std::string* warn, std::string* err);

        bool ReadBuffer(StreamReader* sr, int level, const DeserializeParams& params,
            std::string* warn, std::string* err);

        const std::vector<InternalOrLeafNode>& GetChildNodes() const {
            return child_nodes_;
        }

        std::vector<InternalOrLeafNode>& GetChildNodes() { return child_nodes_; }

        uint32_t GetVoxelSize() const { return uint32_t(value_mask_.DIM); }

    private:
        NodeMask value_mask_;

        // For internal node
        // child nodes are internal or leaf depending on
        // grid_layout_info_[level+1].node_type().
        std::vector<InternalOrLeafNode> child_nodes_;

        NodeMask child_mask_;
        int origin_[3];

        std::vector<ValueType> node_values_;

        // For leaf node

        std::vector<uint8_t> data_;  // Leaf's voxel data.
        uint32_t num_voxels_;
    };

    class RootNode : public Node {
    public:
        RootNode(const GridLayoutInfo& layout_info)
            : Node(layout_info), num_tiles_(0), num_children_(0) {}
        ~RootNode() {}

        /// Deep copy function
        RootNode& Copy(const RootNode& rhs);

        bool ReadTopology(StreamReader* sr, int level, const DeserializeParams& parms,
            std::string* warn, std::string* err);

        bool ReadBuffer(StreamReader* sr, int level, const DeserializeParams& params,
            std::string* warn, std::string* err);

        const std::vector<InternalOrLeafNode>& GetChildNodes() const {
            return child_nodes_;
        }

        std::vector<InternalOrLeafNode>& GetChildNodes() { return child_nodes_; }

        const std::vector<Boundsi>& GetChildBounds() const { return child_bounds_; }

    private:
        // store voxel bounds of child node in global coordinate.
        std::vector<Boundsi> child_bounds_;
        std::vector<InternalOrLeafNode> child_nodes_;

        Value background_;  // Background(region of un-interested area) value
        uint32_t num_tiles_;
        uint32_t num_children_;
    };

    ///
    /// Simple Voxel node.
    /// (integer grid)
    ///
    struct VoxelNode {
        // local bbox
        // must be dividable by each element of `num_divs` for intermediate node.
        uint32_t bmin[3];
        uint32_t bmax[3];

        bool is_leaf;

        uint32_t num_divs[3];  // The number of voxel divisions

        //
        // intermediate(branch)
        //
        double background;  // background value(for empty leaf)

        // offset to child VoxelNode
        // 0 = empty leaf
        std::vector<size_t>
            child_offsets;  // len = num_divs[0] * num_divs[1] * num_divs[2]

        //
        // leaf
        //

        // TODO(syoyo): Support various voxel data type.
        uint32_t num_channels;
        std::vector<float>
            voxels;  // len = num_divs[0] * num_divs[1] * num_divs[2] * num_channels
    };

    class VoxelTree {
    public:
        ///
        /// Returns tree is valid(got success to build tree?)
        ///
        bool Valid();

        ///
        /// Builds Voxel tree from RootNode class
        /// Returns false when failed to build tree(e.g. input `root` is invalid) and
        /// store error message to `err`.
        ///
        bool Build(const RootNode& root, std::string* err);

        ///
        /// Sample voxel value for a given coordinate.
        /// Returns voxel value or background value when `loc` coordinate is empty.
        ///
        /// @param[in] loc Sample coordinate.
        /// @param[in] req_channels Required channels of voxel data.
        /// @param[out] out Sampled voxel value(length = req_channels)
        ///
        void Sample(const uint32_t loc[3], const uint8_t req_channels,
            float* out);

    private:
        // Build tree recursively.
        void BuildTree(const InternalOrLeafNode& root, int depth);

        bool valid_;

        double bmin_[3];   // bounding min of root voxel node(in world coordinate).
        double bmax_[3];   // bounding max of root voxel node(in world coordinate).
        double pitch_[3];  // voxel pitch at leaf level. Assume all voxel has same
                           // pitch size.

        std::vector<VoxelNode> nodes_;  // [0] = root node
    };

#ifdef __clang__
#pragma clang diagnostic pop
#endif

    ///
    /// Parse VDB header from a file.
    /// Returns TINYVDBIO_SUCCESS upon success and `header` will be filled.
    /// Returns false when failed to parse VDB header and store error message to
    /// `err`.
    ///
    VDBStatus ParseVDBHeader(const std::string& filename, VDBHeader* header,
        std::string* err);

    ///
    /// Parse VDB header from memory.
    /// Returns TINYVDBIO_SUCCESS upon success and `header` will be filled.
    /// Returns false when failed to parse VDB header and store error message to
    /// `err`.
    ///
    VDBStatus ParseVDBHeader(const uint8_t* data, const size_t len,
        VDBHeader* header, std::string* err);

    ///
    /// Load Grid descriptors from file
    ///
    /// Returns TINYVDBIO_SUCCESS upon success.
    /// Returns false when failed to read VDB data and store error message to
    /// `err`.
    ///
    VDBStatus ReadGridDescriptors(const std::string& filename,
        const VDBHeader& header,
        std::map<std::string, GridDescriptor>* gd_map,
        std::string* err);

    ///
    /// Load Grid descriptors from memory
    ///
    /// Returns TINYVDBIO_SUCCESS upon success.
    /// Returns false when failed to read VDB data and store error message to
    /// `err`.
    ///
    VDBStatus ReadGridDescriptors(const uint8_t* data, const size_t data_len,
        const VDBHeader& header,
        std::map<std::string, GridDescriptor>* gd_map,
        std::string* err);

    ///
    /// Load Grid data from file
    /// TODO(syoyo): Deprecate
    ///
    /// Returns TINYVDBIO_SUCCESS upon success.
    /// Returns false when failed to read VDB data and store error message to
    /// `err`.
    ///
    VDBStatus ReadGrids(const std::string& filename, const VDBHeader& header,
        const std::map<std::string, GridDescriptor>& gd_map,
        std::string* warn, std::string* err);

    ///
    /// Load Grid data from memory
    ///
    /// Returns TINYVDBIO_SUCCESS upon success.
    /// Returns false when failed to read VDB data and store error message to
    /// `err`.
    /// Returns warning message tot `warn`.
    ///
    VDBStatus ReadGrids(const uint8_t* data, const size_t data_len,
        const VDBHeader& header,
        const std::map<std::string, GridDescriptor>& gd_map,
        std::string* warn, std::string* err);

    ///
    /// Write VDB data to a file.
    ///
    bool SaveVDB(const std::string& filename, std::string* err);

}  // namespace tinyvdb

#endif  // TINY_VDB_IO_H_

#ifdef TINYVDBIO_IMPLEMENTATION

#if !defined(TINYVDBIO_USE_SYSTEM_ZLIB)

#define MINIZ_NO_STDIO
extern "C" {
#include "miniz.h"
}
#else
// Include your zlib.h before including this tinyvdbio.h
#endif

#if defined(TINYVDBIO_USE_BLOSC)
#include <blosc.h>
#endif

#include <iostream>  // HACK
#include <sstream>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#endif

#if defined(_WIN32)


// for MultiByteToWideChar and other UTF8 things.
#if defined(__MINGW32__)
#include <windows.h>
#else
#include <Windows.h>
#endif

#if defined(__GLIBCXX__)  // mingw

#include <fcntl.h>  // _O_RDONLY
#include <ext/stdio_filebuf.h>  // fstream (all sorts of IO stuff) + stdio_filebuf (=streambuf)

#endif

#endif

namespace tinyvdb {



    namespace {

#if defined(_WIN32)

        static inline std::wstring utf8_to_wchar(const std::string& str) {
            int wstr_size =
                MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), nullptr, 0);
            std::wstring wstr(size_t(wstr_size), 0);
            MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), &wstr[0],
                int(wstr.size()));
            return wstr;
        }

#endif

        // TODO(syoyo): Use mmap
        std::vector<uint8_t> read_file_binary(const std::string& filename, std::string* err)
        {
            std::vector<uint8_t> buf;

#if defined(_WIN32)

#if defined(__GLIBCXX__) // mingw gcc
            // Assume system have native UTF-8 suport
            int file_descriptor =
                _wopen(utf8_to_wchar(filename).c_str(), _O_RDONLY | _O_BINARY);
            __gnu_cxx::stdio_filebuf<char> wfile_buf(file_descriptor, std::ios_base::in);
            std::istream f(&wfile_buf);

#elif defined(_MSC_VER) // MSC
            // MSVC extension accepts std::wstring for input filename
            std::ifstream ifs(utf8_to_wchar(filename), std::ifstream::binary);
#else
            // TODO(syoyo): Support UTF-8
            std::ifstream ifs(filename, std::ifstream::binary);
#endif

#else // !WIN32

            // Assume system have native UTF-8 suport
            std::ifstream ifs(filename, std::ifstream::binary);

#endif


            // TODO(syoyo): Use wstring for error message on Win32?
            if (!ifs) {
                if (err) {
                    (*err) = "File not found or cannot open file : " + filename;
                }
                return buf;
            }

            ifs.seekg(0, ifs.end);
            size_t sz = static_cast<size_t>(ifs.tellg());
            if (int64_t(sz) < 0) {
                // Looks reading directory, not a file.
                if (err) {
                    (*err) += "Looks like filename is a directory : \"" + filename + "\"\n";
                }
                return buf;
            }

            if (sz < 16) {
                // ???
                if (err) {
                    (*err) +=
                        "File size too short. Looks like this file is not a VDB : \"" +
                        filename + "\"\n";
                }
                return buf;
            }

            buf.resize(sz);

            ifs.seekg(0, ifs.beg);
            ifs.read(reinterpret_cast<char*>(&buf.at(0)),
                static_cast<std::streamsize>(sz));

            return buf;
        }

    } // namespace local

    std::ostream& operator<<(std::ostream& os, const Boundsi& bound) {
        os << "Boundsi bmin(" << bound.bmin.x << ", " << bound.bmin.y << ", "
            << bound.bmin.z << "), bmax(" << bound.bmax.x << ", " << bound.bmax.y
            << ", " << bound.bmax.z << ")";
        return os;
    }

    const int kOPENVDB_MAGIC = 0x56444220;

    ///
    /// TinyVDBIO's default file version.
    ///
    const uint32_t kTINYVDB_FILE_VERSION = 220;

    // File format versions(identical to OPENVDB_FILE_VERSION_***).
    // This should be same with OpenVDB's implementation.
    // We don't support version less than 220
    enum {
        TINYVDB_FILE_VERSION_SELECTIVE_COMPRESSION = 220,
        TINYVDB_FILE_VERSION_FLOAT_FRUSTUM_BBOX = 221,
        TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION = 222,
        TINYVDB_FILE_VERSION_BLOSC_COMPRESSION = 223,
        TINYVDB_FILE_VERSION_POINT_INDEX_GRID = 223,
        TINYVDB_FILE_VERSION_MULTIPASS_IO = 224
    };

    enum {
        TINYVDB_COMPRESS_NONE = 0,
        TINYVDB_COMPRESS_ZIP = 0x1,
        TINYVDB_COMPRESS_ACTIVE_MASK = 0x2,
        TINYVDB_COMPRESS_BLOSC = 0x4
    };

    // https://gist.github.com/rygorous/2156668
    union FP32LE {
        uint32_t u;
        float f;
        struct {
            uint32_t Mantissa : 23;
            uint32_t Exponent : 8;
            uint32_t Sign : 1;
        } s;
    };

    union FP32BE {
        uint32_t u;
        float f;
        struct {
            uint32_t Sign : 1;
            uint32_t Exponent : 8;
            uint32_t Mantissa : 23;
        } s;
    };

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

    union FP16LE {
        unsigned short u;
        struct {
            uint32_t Mantissa : 10;
            uint32_t Exponent : 5;
            uint32_t Sign : 1;
        } s;
    };

    union FP16BE {
        unsigned short u;
        struct {
            uint32_t Sign : 1;
            uint32_t Exponent : 5;
            uint32_t Mantissa : 10;
        } s;
    };

#ifdef __clang__
#pragma clang diagnostic pop
#endif

    static inline FP32LE half_to_float_le(FP16LE h) {
        static const FP32LE magic = { 113 << 23 };
        static const uint32_t shifted_exp = 0x7c00
            << 13;  // exponent mask after shift
        FP32LE o;

        o.u = (h.u & 0x7fffU) << 13U;       // exponent/mantissa bits
        uint32_t exp_ = shifted_exp & o.u;  // just the exponent
        o.u += (127 - 15) << 23;            // exponent adjust

        // handle exponent special cases
        if (exp_ == shifted_exp)    // Inf/NaN?
            o.u += (128 - 16) << 23;  // extra exp adjust
        else if (exp_ == 0)         // Zero/Denormal?
        {
            o.u += 1 << 23;  // extra exp adjust
            o.f -= magic.f;  // renormalize
        }

        o.u |= (h.u & 0x8000U) << 16U;  // sign bit
        return o;
    }

    static inline FP16LE float_to_half_full_le(FP32LE f) {
        FP16LE o = { 0 };

        // Based on ISPC reference code (with minor modifications)
        if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
            o.s.Exponent = 0;
        else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
        {
            o.s.Exponent = 31;
            o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
        }
        else                                      // Normalized number
        {
            // Exponent unbias the single, then bias the halfp
            int newexp = f.s.Exponent - 127 + 15;
            if (newexp >= 31)  // Overflow, return signed infinity
                o.s.Exponent = 31;
            else if (newexp <= 0)  // Underflow
            {
                if ((14 - newexp) <= 24)  // Mantissa might be non-zero
                {
                    uint32_t mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
                    o.s.Mantissa = mant >> (14 - newexp);
                    if ((mant >> (13 - newexp)) & 1)  // Check for rounding
                        o.u++;  // Round, might overflow into exp bit, but this is OK
                }
            }
            else {
                o.s.Exponent = static_cast<uint32_t>(newexp);
                o.s.Mantissa = f.s.Mantissa >> 13;
                if (f.s.Mantissa & 0x1000)  // Check for rounding
                    o.u++;                    // Round, might overflow to inf, this is OK
            }
        }

        o.s.Sign = f.s.Sign;
        return o;
    }

    static inline FP32BE half_to_float_be(FP16BE h) {
        static const FP32BE magic = { 113 << 23 };
        static const uint32_t shifted_exp = 0x7c00
            << 13;  // exponent mask after shift
        FP32BE o;

        o.u = (h.u & 0x7fffU) << 13U;       // exponent/mantissa bits
        uint32_t exp_ = shifted_exp & o.u;  // just the exponent
        o.u += (127 - 15) << 23;            // exponent adjust

        // handle exponent special cases
        if (exp_ == shifted_exp)    // Inf/NaN?
            o.u += (128 - 16) << 23;  // extra exp adjust
        else if (exp_ == 0)         // Zero/Denormal?
        {
            o.u += 1 << 23;  // extra exp adjust
            o.f -= magic.f;  // renormalize
        }

        o.u |= (h.u & 0x8000U) << 16U;  // sign bit
        return o;
    }

    static inline FP16BE float_to_half_full_le(FP32BE f) {
        FP16BE o = { 0 };

        // Based on ISPC reference code (with minor modifications)
        if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
            o.s.Exponent = 0;
        else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
        {
            o.s.Exponent = 31;
            o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
        }
        else                                      // Normalized number
        {
            // Exponent unbias the single, then bias the halfp
            int newexp = f.s.Exponent - 127 + 15;
            if (newexp >= 31)  // Overflow, return signed infinity
                o.s.Exponent = 31;
            else if (newexp <= 0)  // Underflow
            {
                if ((14 - newexp) <= 24)  // Mantissa might be non-zero
                {
                    uint32_t mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
                    o.s.Mantissa = mant >> (14 - newexp);
                    if ((mant >> (13 - newexp)) & 1)  // Check for rounding
                        o.u++;  // Round, might overflow into exp bit, but this is OK
                }
            }
            else {
                o.s.Exponent = static_cast<uint32_t>(newexp);
                o.s.Mantissa = f.s.Mantissa >> 13;
                if (f.s.Mantissa & 0x1000)  // Check for rounding
                    o.u++;                    // Round, might overflow to inf, this is OK
            }
        }

        o.s.Sign = f.s.Sign;
        return o;
    }

    static inline void swap2(unsigned short* val) {
        unsigned short tmp = *val;
        uint8_t* dst = reinterpret_cast<uint8_t*>(val);
        uint8_t* src = reinterpret_cast<uint8_t*>(&tmp);

        dst[0] = src[1];
        dst[1] = src[0];
    }

    static inline void swap4(uint32_t* val) {
        uint32_t tmp = *val;
        uint8_t* dst = reinterpret_cast<uint8_t*>(val);
        uint8_t* src = reinterpret_cast<uint8_t*>(&tmp);

        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }

    static inline void swap4(int* val) {
        int tmp = *val;
        uint8_t* dst = reinterpret_cast<uint8_t*>(val);
        uint8_t* src = reinterpret_cast<uint8_t*>(&tmp);

        dst[0] = src[3];
        dst[1] = src[2];
        dst[2] = src[1];
        dst[3] = src[0];
    }

    static inline void swap8(uint64_t* val) {
        uint64_t tmp = (*val);
        uint8_t* dst = reinterpret_cast<uint8_t*>(val);
        uint8_t* src = reinterpret_cast<uint8_t*>(&tmp);

        dst[0] = src[7];
        dst[1] = src[6];
        dst[2] = src[5];
        dst[3] = src[4];
        dst[4] = src[3];
        dst[5] = src[2];
        dst[6] = src[1];
        dst[7] = src[0];
    }

    static inline void swap8(int64_t* val) {
        int64_t tmp = (*val);
        uint8_t* dst = reinterpret_cast<uint8_t*>(val);
        uint8_t* src = reinterpret_cast<uint8_t*>(&tmp);

        dst[0] = src[7];
        dst[1] = src[6];
        dst[2] = src[5];
        dst[3] = src[4];
        dst[4] = src[3];
        dst[5] = src[2];
        dst[6] = src[1];
        dst[7] = src[0];
    }

    ///
    /// Simple stream reader
    ///
    class StreamReader {
    public:
        explicit StreamReader(const uint8_t* binary, const size_t length,
            const bool swap_endian)
            : binary_(binary), length_(length), swap_endian_(swap_endian), idx_(0) {
            (void)pad_;
        }

        bool seek_set(const uint64_t offset) {
            if (offset > length_) {
                return false;
            }

            idx_ = offset;
            return true;
        }

        bool seek_from_currect(const int64_t offset) {
            if ((int64_t(idx_) + offset) < 0) {
                return false;
            }

            if (size_t((int64_t(idx_) + offset)) > length_) {
                return false;
            }

            idx_ = size_t(int64_t(idx_) + offset);
            return true;
        }

        size_t read(const size_t n, const uint64_t dst_len, uint8_t* dst) {
            size_t len = n;
            if ((idx_ + len) > length_) {
                len = length_ - idx_;
            }

            if (len > 0) {
                if (dst_len < len) {
                    // dst does not have enough space. return 0 for a while.
                    return 0;
                }

                memcpy(dst, &binary_[idx_], len);
                idx_ += len;
                return len;

            }
            else {
                return 0;
            }
        }

        bool read1(uint8_t* ret) {
            if ((idx_ + 1) > length_) {
                return false;
            }

            const uint8_t val = binary_[idx_];

            (*ret) = val;
            idx_ += 1;

            return true;
        }

        bool read_bool(bool* ret) {
            if ((idx_ + 1) > length_) {
                return false;
            }

            const char val = static_cast<const char>(binary_[idx_]);

            (*ret) = bool(val);
            idx_ += 1;

            return true;
        }

        bool read1(char* ret) {
            if ((idx_ + 1) > length_) {
                return false;
            }

            const char val = static_cast<const char>(binary_[idx_]);

            (*ret) = val;
            idx_ += 1;

            return true;
        }

        bool read2(unsigned short* ret) {
            if ((idx_ + 2) > length_) {
                return false;
            }

            unsigned short val =
                *(reinterpret_cast<const unsigned short*>(&binary_[idx_]));

            if (swap_endian_) {
                swap2(&val);
            }

            (*ret) = val;
            idx_ += 2;

            return true;
        }

        bool read4(uint32_t* ret) {
            if ((idx_ + 4) > length_) {
                return false;
            }

            uint32_t val = *(reinterpret_cast<const uint32_t*>(&binary_[idx_]));

            if (swap_endian_) {
                swap4(&val);
            }

            (*ret) = val;
            idx_ += 4;

            return true;
        }

        bool read4(int* ret) {
            if ((idx_ + 4) > length_) {
                return false;
            }

            int val = *(reinterpret_cast<const int*>(&binary_[idx_]));

            if (swap_endian_) {
                swap4(&val);
            }

            (*ret) = val;
            idx_ += 4;

            return true;
        }

        bool read8(uint64_t* ret) {
            if ((idx_ + 8) > length_) {
                return false;
            }

            uint64_t val = *(reinterpret_cast<const uint64_t*>(&binary_[idx_]));

            if (swap_endian_) {
                swap8(&val);
            }

            (*ret) = val;
            idx_ += 8;

            return true;
        }

        bool read8(int64_t* ret) {
            if ((idx_ + 8) > length_) {
                return false;
            }

            int64_t val = *(reinterpret_cast<const int64_t*>(&binary_[idx_]));

            if (swap_endian_) {
                swap8(&val);
            }

            (*ret) = val;
            idx_ += 8;

            return true;
        }

        bool read_float(float* ret) {
            if (!ret) {
                return false;
            }

            float value;
            if (!read4(reinterpret_cast<int*>(&value))) {
                return false;
            }

            (*ret) = value;

            return true;
        }

        bool read_double(double* ret) {
            if (!ret) {
                return false;
            }

            double value;
            if (!read8(reinterpret_cast<uint64_t*>(&value))) {
                return false;
            }

            (*ret) = value;

            return true;
        }

        bool read_value(Value* inout) {
            if (!inout) {
                return false;
            }

            if (inout->Type() == VALUE_TYPE_FLOAT) {
                float value;
                if (!read_float(&value)) {
                    return false;
                }

                (*inout) = Value(value);
            }
            else if (inout->Type() == VALUE_TYPE_INT) {
                int value;
                if (!read4(&value)) {
                    return false;
                }

                (*inout) = Value(value);
            }
            else {
                TINYVDBIO_ASSERT(0);
                return false;
            }

            return true;
        }

        size_t tell() const { return idx_; }

        const uint8_t* data() const { return binary_; }

        bool swap_endian() const { return swap_endian_; }

        size_t size() const { return length_; }

    private:
        const uint8_t* binary_;
        const size_t length_;
        bool swap_endian_;
        char pad_[7];
        uint64_t idx_;
    };

    static Value ReadValue(StreamReader* sr, const ValueType type) {
        if (type == VALUE_TYPE_NULL) {
            return Value();
        }
        else if (type == VALUE_TYPE_BOOL) {
            char value;
            sr->read1(&value);
            return Value(value);
        }
        else if (type == VALUE_TYPE_FLOAT) {
            float value;
            sr->read_float(&value);
            return Value(value);
        }
        else if (type == VALUE_TYPE_INT) {
            int value;
            sr->read4(&value);
            return Value(value);
        }
        else if (type == VALUE_TYPE_DOUBLE) {
            double value;
            sr->read_double(&value);
            return Value(value);
        }
        // ???
        return Value();
    }

    struct DeserializeParams {
        uint32_t file_version;
        uint32_t compression_flags;
        bool half_precision;
        char __pad__[7];
        Value background;
    };

    static inline std::string ReadString(StreamReader* sr) {
        uint32_t size = 0;
        sr->read4(&size);
        if (size > 0) {
            std::string buffer(size, ' ');
            sr->read(size, size, reinterpret_cast<uint8_t*>(&buffer[0]));
            return buffer;
        }
        return std::string();
    }

    static inline void WriteString(std::ostream& os, const std::string& name) {
        uint32_t size = static_cast<uint32_t>(name.size());
        os.write(reinterpret_cast<char*>(&size), sizeof(uint32_t));
        os.write(&name[0], size);
    }

    static inline bool ReadMetaBool(StreamReader* sr) {
        char c = 0;
        uint32_t size;
        sr->read4(&size);
        if (size == 1) {
            sr->read(1, 1, reinterpret_cast<uint8_t*>(&c));
        }
        return bool(c);
    }

    static inline float ReadMetaFloat(StreamReader* sr) {
        float f = 0.0f;
        uint32_t size;
        sr->read4(&size);
        if (size == sizeof(float)) {
            sr->read4(reinterpret_cast<uint32_t*>(&f));
        }
        return f;
    }

    static inline void ReadMetaVec3i(StreamReader* sr, int v[3]) {
        uint32_t size;
        sr->read4(&size);
        if (size == 3 * sizeof(int)) {
            sr->read4(&v[0]);
            sr->read4(&v[1]);
            sr->read4(&v[2]);
        }
    }

    static inline void ReadMetaVec3d(StreamReader* sr, double v[3]) {
        uint32_t size;
        sr->read4(&size);
        if (size == 3 * sizeof(double)) {
            sr->read_double(&v[0]);
            sr->read_double(&v[1]);
            sr->read_double(&v[2]);
        }
    }

    static inline int64_t ReadMetaInt64(StreamReader* sr) {
        uint32_t size;
        int64_t i64 = 0;
        sr->read4(&size);
        if (size == sizeof(int64_t)) {
            sr->read8(reinterpret_cast<uint64_t*>(&i64));
        }
        return i64;
    }

    static inline void ReadVec3d(StreamReader* sr, double v[3]) {
        sr->read_double(&v[0]);
        sr->read_double(&v[1]);
        sr->read_double(&v[2]);
    }

    // https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
    static inline bool EndsWidth(std::string const& value,
        std::string const& ending) {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

#if 0
    template <std::size_t N>
    bool BitMask<N>::load(StreamReader* sr) {
        std::vector<uint8_t> buf(mask_.size() / 8);

        sr->read(mask_.size(), mask_.size(), buf.data());

        // Reconstruct bit mask
        // TODO(syoyo): endian
        for (size_t j = 0; j < mask_.size() / 8; j++) {
            for (size_t i = 0; i < 8; i++) {
                uint8_t bit = (buf[j] >> i) & 0x1;
                mask_.set(j * 8 + i, bit);
            }
        }

        return true;
    }
#endif

    static bool DecompressZip(uint8_t* dst,
        size_t* uncompressed_size /* inout */,
        const uint8_t* src, size_t src_size) {
        if ((*uncompressed_size) == src_size) {
            // Data is not compressed.
            memcpy(dst, src, src_size);
            return true;
        }
        std::vector<uint8_t> tmpBuf(*uncompressed_size);

#if defined(TINYVDBIO_USE_SYSTEM_ZLIB)
        int ret = uncompress(&tmpBuf.at(0), uncompressed_size, src, src_size);
        if (Z_OK != ret) {
            return false;
        }
#else
        mz_ulong sz = static_cast<mz_ulong>(*uncompressed_size); // 32bit in Win64 llvm-mingw(clang)
        int ret = mz_uncompress(&tmpBuf.at(0), &sz, src, static_cast<mz_ulong>(src_size));
        if (MZ_OK != ret) {
            return false;
        }
        (*uncompressed_size) = size_t(sz);
#endif

        memcpy(dst, tmpBuf.data(), (*uncompressed_size));

        return true;
    }

#if defined(TINYVDBIO_USE_BLOSC)
    static bool DecompressBlosc(uint8_t* dst, size_t uncompressed_size,
        const uint8_t* src, size_t src_size) {
        if (uncompressed_size == src_size) {
            // Data is not compressed.
            memcpy(dst, src, src_size);
            return true;
        }

        std::cout << "DBG: uncompressed_size = " << uncompressed_size << ", src_size = " << src_size << std::endl;
        const int numUncompressedBytes = blosc_decompress_ctx(
            /*src=*/src, /*dest=*/dst, src_size, /*numthreads=*/1);

        std::cout << "DBG: numUncompressedBytes = " << numUncompressedBytes << ", src_size = " << src_size << std::endl;

        if (numUncompressedBytes < 1) {
            // TODO(syoyo): print warning.
            //
            // numUncompressedBytes may be 0 for small dataset(e.g. <= 16bytes), so 0 or negative may be accepted.
        }

        if (numUncompressedBytes != int(uncompressed_size)) {
            std::cout << "aaa" << std::endl;
            return false;
        }

        return true;
    }
#endif

    static bool ReadAndDecompressData(StreamReader* sr, uint8_t* dst_data,
        size_t element_size, size_t count,
        uint32_t compression_mask, std::string* warn,
        std::string* err) {
        (void)warn;

        if (compression_mask & TINYVDB_COMPRESS_BLOSC) {
            std::cout << "HACK: BLOSLC" << std::endl;

#if defined(TINYVDBIO_USE_BLOSC)
            // Read the size of the compressed data.
            // A negative size indicates uncompressed data.
            int64_t numCompressedBytes;
            sr->read8(&numCompressedBytes);

            std::cout << "numCompressedBytes " << numCompressedBytes << std::endl;
            if (numCompressedBytes <= 0) {
                if (dst_data == NULL) {
                    // seek over this data.
                    sr->seek_set(sr->tell() + element_size * count);
                }
                else {
                    sr->read(element_size * count, element_size * count, dst_data);
                }
            }
            else {
                size_t uncompressed_size = element_size * count;
                std::vector<uint8_t> buf;
                buf.resize(size_t(numCompressedBytes));

                if (!sr->read(size_t(numCompressedBytes), size_t(numCompressedBytes),
                    buf.data())) {
                    if (err) {
                        (*err) +=
                            "Failed to read num compressed bytes in ReadAndDecompressData.\n";
                    }
                    return false;
                }

                if (!DecompressBlosc(dst_data, uncompressed_size, buf.data(),
                    size_t(numCompressedBytes))) {
                    if (err) {
                        (*err) += "Failed to decode BLOSC data.\n";
                    }
                    return false;
                }
            }

            std::cout << "blosc decode ok" << std::endl;

#else
            std::cout << "HACK: BLOSLC is TODO" << std::endl;
            // TODO(syoyo):
            if (err) {
                (*err) += "Decoding BLOSC is not supported in this build.\n";
            }
            return false;
#endif
        }
        else if (compression_mask & TINYVDB_COMPRESS_ZIP) {
            // Read the size of the compressed data.
            // A negative size indicates uncompressed data.
            int64_t numZippedBytes;
            sr->read8(&numZippedBytes);
            std::cout << "numZippedBytes = " << numZippedBytes << std::endl;

            if (numZippedBytes <= 0) {
                if (dst_data == NULL) {
                    // seek over this data.
                    sr->seek_set(sr->tell() + element_size * count);
                }
                else {
                    sr->read(element_size * count, element_size * count, dst_data);
                }
            }
            else {
                size_t uncompressed_size = element_size * count;
                std::vector<uint8_t> buf;
                buf.resize(size_t(numZippedBytes));

                if (!sr->read(size_t(numZippedBytes), size_t(numZippedBytes),
                    buf.data())) {
                    if (err) {
                        (*err) +=
                            "Failed to read num zipped bytes in ReadAndDecompressData.\n";
                    }
                    return false;
                }

                if (!DecompressZip(dst_data, &uncompressed_size, buf.data(),
                    size_t(numZippedBytes))) {
                    if (err) {
                        (*err) += "Failed to decode zip data.\n";
                    }
                    return false;
                }
            }
        }
        else {
            std::cout << "HACK: uncompressed" << std::endl;
            std::cout << "  elem_size = " << element_size << ", count = " << count
                << std::endl;
            if (dst_data == NULL) {
                // seek over this data.
                sr->seek_set(sr->tell() + element_size * count);
            }
            else {
                sr->read(element_size * count, element_size * count, dst_data);
            }
        }

        if (sr->swap_endian()) {
            if (element_size == 2) {
                unsigned short* ptr = reinterpret_cast<unsigned short*>(dst_data);
                for (size_t i = 0; i < count; i++) {
                    swap2(ptr + i);
                }
            }
            else if (element_size == 4) {
                uint32_t* ptr = reinterpret_cast<uint32_t*>(dst_data);
                for (size_t i = 0; i < count; i++) {
                    swap4(ptr + i);
                }
            }
            else if (element_size == 8) {
                uint64_t* ptr = reinterpret_cast<uint64_t*>(dst_data);
                for (size_t i = 0; i < count; i++) {
                    swap8(ptr + i);
                }
            }
        }

        return true;
    }

    static bool ReadValues(StreamReader* sr, const uint32_t compression_flags,
        size_t num_values, ValueType value_type,
        std::vector<uint8_t>* values, std::string* warn,
        std::string* err) {
        // usually fp16 or fp32
        TINYVDBIO_ASSERT((value_type == VALUE_TYPE_FLOAT) ||
            (value_type == VALUE_TYPE_HALF));

        // Advance stream position when destination buffer is null.
        const bool seek = (values == NULL);

        // read data.
        if (seek) {
            // should not be 'seek' at the monent.
            TINYVDBIO_ASSERT(0);
        }
        else {
            if (!ReadAndDecompressData(sr, values->data(), GetValueTypeSize(value_type),
                num_values, compression_flags, warn, err)) {
                return false;
            }
        }

        return true;
    }

    static bool ReadMaskValues(StreamReader* sr, const uint32_t compression_flags,
        const uint32_t file_version, const Value background,
        size_t num_values, ValueType value_type,
        NodeMask value_mask,
        std::vector<uint8_t>* values,
        std::string* warn, std::string* err) {
        // Advance stream position when destination buffer is null.
        const bool seek = (values == NULL);

        const bool mask_compressed = compression_flags & TINYVDB_COMPRESS_ACTIVE_MASK;

        char per_node_flag = NO_MASK_AND_ALL_VALS;
        if (file_version >= TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
            if (seek && !mask_compressed) {
                // selection mask and/or inactive value is saved.
                sr->seek_set(sr->tell() + 1);  // advance 1 byte.
            }
            else {
                sr->read1(&per_node_flag);
            }
        }

        Value inactiveVal1 = background;
        Value inactiveVal0 =
            ((per_node_flag == NO_MASK_OR_INACTIVE_VALS) ? background
                : Negate(background));

        if (per_node_flag == NO_MASK_AND_ONE_INACTIVE_VAL ||
            per_node_flag == MASK_AND_ONE_INACTIVE_VAL ||
            per_node_flag == MASK_AND_TWO_INACTIVE_VALS) {
            // inactive val
            if (seek) {
                sr->seek_set(sr->tell() + sizeof(inactiveVal0));
            }
            else {
                sr->read_value(&inactiveVal0);
            }

            if (per_node_flag == MASK_AND_TWO_INACTIVE_VALS) {
                // Read the second of two distinct inactive values.
                if (seek) {
                    sr->seek_set(sr->tell() + inactiveVal1.Size());
                }
                else {
                    sr->read_value(&inactiveVal1);
                }
            }
        }

        NodeMask selection_mask(value_mask.LOG2DIM);
        if (per_node_flag == MASK_AND_NO_INACTIVE_VALS ||
            per_node_flag == MASK_AND_ONE_INACTIVE_VAL ||
            per_node_flag == MASK_AND_TWO_INACTIVE_VALS) {
            // For use in mask compression (only), read the bitmask that selects
            // between two distinct inactive values.
            if (seek) {
                sr->seek_set(sr->tell() + uint32_t(selection_mask.memUsage()));
            }
            else {
                selection_mask.load(sr);
            }
        }

        size_t read_count = num_values;

        if (mask_compressed && per_node_flag != NO_MASK_AND_ALL_VALS &&
            file_version >= TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
            read_count = size_t(value_mask.count_on());
            std::cout << "3 metadata. read count = " << read_count << std::endl;
        }

        std::cout << "read num = " << read_count << std::endl;

        std::vector<uint8_t> tmp_buf(read_count * GetValueTypeSize(value_type));

        // Read mask data.
        if (!ReadAndDecompressData(sr, tmp_buf.data(), GetValueTypeSize(value_type),
            size_t(read_count), compression_flags, warn,
            err)) {
            return false;
        }

        // If mask compression is enabled and the number of active values read into
        // the temp buffer is smaller than the size of the destination buffer,
        // then there are missing (inactive) values.
        if (!seek && mask_compressed && read_count != num_values) {
            // Restore inactive values, using the background value and, if available,
            // the inside/outside mask.  (For fog volumes, the destination buffer is
            // assumed to be initialized to background value zero, so inactive values
            // can be ignored.)
            size_t sz = GetValueTypeSize(value_type);
            for (size_t destIdx = 0, tempIdx = 0;
                destIdx < size_t(selection_mask.BITSIZE); ++destIdx) {
                if (value_mask.is_on(int32_t(destIdx))) {
                    // Copy a saved active value into this node's buffer.
                    memcpy(&values->at(destIdx * sz), &tmp_buf.at(tempIdx * sz), sz);
                    ++tempIdx;
                }
                else {
                    // Reconstruct an unsaved inactive value and copy it into this node's
                    // buffer.
                    if (selection_mask.is_on(int32_t(destIdx))) {
                        memcpy(&values->at(destIdx * sz), &inactiveVal1, sz);
                    }
                    else {
                        memcpy(&values->at(destIdx * sz), &inactiveVal0, sz);
                    }
                }
            }
        }
        else {
            memcpy(values->data(), tmp_buf.data(),
                num_values * GetValueTypeSize(value_type));
        }

        return true;
    }

    bool NodeMask::load(StreamReader* sr) {
        // std::cout << "DBG: mWords size = " << this->size() << std::endl;
        // std::cout << "DBG: mWords.size = " << mWords.size() << std::endl;

        bool ret = sr->read(this->memUsage(), this->memUsage(), bits.data());

        return ret;
    }

    bool RootNode::ReadTopology(StreamReader* sr, int level,
        const DeserializeParams& params, std::string* warn,
        std::string* err) {
        std::cout << "Root background loc " << sr->tell() << std::endl;

        // Read background value;
        background_ =
            ReadValue(sr, grid_layout_info_.GetNodeInfo(level).value_type());

        std::cout << "background : " << background_ << ", size = "
            << GetValueTypeSize(
                grid_layout_info_.GetNodeInfo(level).value_type())
            << std::endl;

        sr->read4(&num_tiles_);
        sr->read4(&num_children_);

        if ((num_tiles_ == 0) && (num_children_ == 0)) {
            return false;
        }

        std::cout << "num_tiles " << num_tiles_ << std::endl;
        std::cout << "num_children " << num_children_ << std::endl;

        // Read tiles.
        for (uint32_t n = 0; n < num_tiles_; n++) {
            int vec[3];
            Value value;
            bool active;

            sr->read4(&vec[0]);
            sr->read4(&vec[1]);
            sr->read4(&vec[2]);
            value = ReadValue(sr, grid_layout_info_.GetNodeInfo(level).value_type());
            sr->read_bool(&active);

            std::cout << "[" << n << "] vec = (" << vec[0] << ", " << vec[1] << ", "
                << vec[2] << "), value = " << value << ", active = " << active
                << std::endl;
        }

        // Read child nodes.
        for (uint32_t n = 0; n < num_children_; n++) {
            Vec3i coord;
            sr->read4(&coord.x);
            sr->read4(&coord.y);
            sr->read4(&coord.z);

            // Child should be InternalOrLeafNode type
            TINYVDBIO_ASSERT((level + 1) < grid_layout_info_.NumLevels());
            TINYVDBIO_ASSERT(grid_layout_info_.GetNodeInfo(level + 1).node_type() ==
                NODE_TYPE_INTERNAL);

            InternalOrLeafNode child_node(grid_layout_info_);
            if (!child_node.ReadTopology(sr, /* level */ level + 1, params, warn,
                err)) {
                return false;
            }

            child_nodes_.push_back(child_node);

            std::cout << "root.child[" << n << "] vec = (" << coord.x << ", " << coord.y
                << ", " << coord.z << std::endl;

            uint32_t global_voxel_size = grid_layout_info_.ComputeGlobalVoxelSize(0);
            Boundsi bounds;
            bounds.bmin = coord;
            bounds.bmax.x = coord.x + int(global_voxel_size);
            bounds.bmax.y = coord.y + int(global_voxel_size);
            bounds.bmax.z = coord.y + int(global_voxel_size);
            child_bounds_.push_back(bounds);
        }

        // HACK
        {
            VoxelTree tree;
            std::string _err;

            bool ret = tree.Build(*this, &_err);
            if (!_err.empty()) {
                std::cerr << _err << std::endl;
            }
            if (!ret) {
                return false;
            }
        }

        return true;
    }

    bool RootNode::ReadBuffer(StreamReader* sr, int level,
        const DeserializeParams& params, std::string* warn,
        std::string* err) {
        std::cout << "root readbuffer pos " << sr->tell() << std::endl;

        // Recursive call
        for (size_t i = 0; i < num_children_; i++) {
            if (!child_nodes_[i].ReadBuffer(sr, level + 1, params, warn, err)) {
                if (err) {
                    (*err) += "Failed to read buffer.\n";
                }
                return false;
            }
            std::cout << "ReadBuffer done. child_node[" << i << "]" << std::endl;
        }

        return true;
    }

    bool InternalOrLeafNode::ReadTopology(StreamReader* sr, const int level,
        const DeserializeParams& params,
        std::string* warn, std::string* err) {
        (void)params;

        int node_type = grid_layout_info_.GetNodeInfo(level).node_type();

        if (node_type == NODE_TYPE_INTERNAL) {
#if 0  // API3
            {

                int buffer_count;
                sr->read4(&buffer_count);
                if (buffer_count != 1) {
                    // OPENVDB_LOG_WARN("multi-buffer trees are no longer supported");
                }
            }
#endif

            std::cout << "topo: buffer count offt = " << sr->tell() << std::endl;
            std::cout << "readtopo: level = " << level << std::endl;

            child_mask_.Alloc(grid_layout_info_.GetNodeInfo(level).log2dim());
            child_mask_.load(sr);
            std::cout << "topo: child mask buffer count offt = " << sr->tell()
                << std::endl;

            value_mask_.Alloc(grid_layout_info_.GetNodeInfo(level).log2dim());
            value_mask_.load(sr);
            std::cout << "topo: value mask buffer count offt = " << sr->tell()
                << std::endl;

            const bool old_version =
                params.file_version < TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION;

            int NUM_VALUES =
                1 << (3 *
                    grid_layout_info_.GetNodeInfo(level)
                    .log2dim());  // total voxel count represented by this node

            std::cout << "num value_mask = " << value_mask_.BITSIZE << std::endl;
            std::cout << "NUM_VALUES = " << NUM_VALUES << std::endl;

            // Older version will have less values
            const int num_values =
                (old_version ? int(child_mask_.nbits() - child_mask_.count_on())
                    : NUM_VALUES);

            {
                std::vector<uint8_t> values;
                values.resize(
                    GetValueTypeSize(grid_layout_info_.GetNodeInfo(level).value_type()) *
                    size_t(num_values));

                if (!ReadMaskValues(sr, params.compression_flags, params.file_version,
                    params.background, size_t(num_values),
                    grid_layout_info_.GetNodeInfo(level).value_type(),
                    value_mask_, &values, warn, err)) {
                    if (err) {
                        std::stringstream ss;
                        ss << "Failed to read mask values in ReadTopology. level = " << level
                            << std::endl;
                        (*err) += ss.str();
                    }

                    return false;
                }

                // Copy values from the array into this node's table.
                if (old_version) {
                    TINYVDBIO_ASSERT(size_t(num_values) <= node_values_.size());

                    // loop over child flags is off.
                    int n = 0;
                    for (int32_t i = 0; i < int32_t(NUM_VALUES); i++) {
                        if (child_mask_.is_off(i)) {
                            // mNodes[iter.pos()].setValue(values[n++]);
                            n++;
                        }
                    }
                    TINYVDBIO_ASSERT(n == num_values);
                }
                else {
                    // loop over child flags is off.
                    for (int32_t i = 0; i < int32_t(NUM_VALUES); i++) {
                        if (child_mask_.is_off(i)) {
                            // mNodes[iter.pos()].setValue(values[iter.pos());
                        }
                    }
                }
            }

            std::cout << "SIZE = " << child_mask_.BITSIZE << std::endl;
            child_nodes_.resize(size_t(child_mask_.BITSIZE), grid_layout_info_);

            // loop over child node
            for (int32_t i = 0; i < child_mask_.BITSIZE; i++) {
                if (child_mask_.is_on(i)) {
                    // if (node_desc_.child_node_desc_) {
                    if (1) {  // HACK
                        TINYVDBIO_ASSERT(i < int32_t(child_nodes_.size()));
                        // child_nodes_[i] = new InternalOrLeafNode
                        if (!child_nodes_[size_t(i)].ReadTopology(sr, level + 1, params, warn,
                            err)) {
                            return false;
                        }
                    }
                    else {  // leaf
                   // TODO: add to child.
                        TINYVDBIO_ASSERT(0);
                    }
                }
            }

            return true;

        }
        else {  // leaf

            value_mask_.Alloc(grid_layout_info_.GetNodeInfo(level).log2dim());
            bool ret = value_mask_.load(sr);

            if (!ret) {
                if (err) {
                    (*err) += "Failed to load value mask in leaf node.\n";
                }
                return false;
            }

            num_voxels_ = 1 << (3 * grid_layout_info_.GetNodeInfo(level).log2dim());

            return true;
        }
    }

    bool InternalOrLeafNode::ReadBuffer(StreamReader* sr, int level,
        const DeserializeParams& params,
        std::string* warn, std::string* err) {
        int node_type = grid_layout_info_.GetNodeInfo(level).node_type();
        std::cout << "internalOrLeaf : read buffer" << std::endl;

        if (node_type == NODE_TYPE_INTERNAL) {
            size_t count = 0;
            for (int32_t i = 0; i < child_mask_.BITSIZE; i++) {
                if (child_mask_.is_on(i)) {
                    std::cout << "InternalOrLeafNode.ReadBuffer[" << count << "]"
                        << std::endl;
                    // TODO: FIXME
                    if (!child_nodes_[size_t(i)].ReadBuffer(sr, level + 1, params, warn,
                        err)) {
                        return false;
                    }
                    count++;
                }
            }

            return true;

        }
        else {  // leaf
            TINYVDBIO_ASSERT(node_type == NODE_TYPE_LEAF);
            char num_buffers = 1;

            std::cout << "LeafNode.ReadBuffer pos = " << sr->tell() << std::endl;
            std::cout << " value_mask_.size = " << value_mask_.memUsage() << std::endl;

            std::cout << " value_mask.bits = " << value_mask_.bitString() << "\n";

            // Seek over the value mask.
            sr->seek_from_currect(int64_t(value_mask_.memUsage()));

            std::cout << "is pos = " << sr->tell() << std::endl;

            if (params.file_version < TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
                int coord[3];

                // Read coordinate origin and num buffers.
                sr->read4(&coord[0]);
                sr->read4(&coord[1]);
                sr->read4(&coord[2]);

                sr->read1(&num_buffers);
                TINYVDBIO_ASSERT(num_buffers == 1);
            }

            const bool mask_compressed =
                params.compression_flags & TINYVDB_COMPRESS_ACTIVE_MASK;

            const bool seek = false;

            char per_node_flag = NO_MASK_AND_ALL_VALS;
            if (params.file_version >= TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
                if (seek && !mask_compressed) {
                    // selection mask and/or inactive value is saved.
                    sr->seek_set(sr->tell() + 1);  // advance 1 byte.
                }
                else {
                    sr->read1(&per_node_flag);
                }
            }

            // TODO(syoyo): clipBBox check.

            TINYVDBIO_ASSERT(num_voxels_ > 0);

            size_t read_count = num_voxels_;

            if (mask_compressed && per_node_flag != NO_MASK_AND_ALL_VALS &&
                (params.file_version >= TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION)) {
                read_count = size_t(value_mask_.count_on());
                std::cout << "value_mask_.count_on = " << read_count << std::endl;
            }

            std::cout << "read_count = " << read_count << std::endl;

            data_.resize(
                read_count *
                GetValueTypeSize(grid_layout_info_.GetNodeInfo(level).value_type()));

            std::cout << "data.size = " << data_.size() << "\n";

            // Data is tightly packed and compressed.
            bool ret = ReadValues(sr, params.compression_flags, read_count,
                grid_layout_info_.GetNodeInfo(level).value_type(),
                &data_, warn, err);

            // HACK
            if (4 == GetValueTypeSize(grid_layout_info_.GetNodeInfo(level).value_type())) {
                const float* ptr = reinterpret_cast<const float*>(data_.data());
                for (size_t i = 0; i < read_count; i++) {
                    std::cout << "[" << i << "] = " << ptr[i] << "\n";
                }
            }

            return ret;
        }
    }

    GridDescriptor::GridDescriptor()
        : save_float_as_half_(false),
        grid_byte_offset_(0),
        block_byte_offset_(0),
        end_byte_offset_(0) {}

    GridDescriptor::GridDescriptor(const std::string& name,
        const std::string& grid_type,
        bool save_float_as_half)
        : grid_name_(StripSuffix(name)),
        unique_name_(name),
        grid_type_(grid_type),
        save_float_as_half_(save_float_as_half),
        grid_byte_offset_(0),
        block_byte_offset_(0),
        end_byte_offset_(0) {}

    // GridDescriptor::GridDescriptor(const GridDescriptor &rhs) {
    //}

    // GridDescriptor::~GridDescriptor() {}

    std::string GridDescriptor::AddSuffix(const std::string& name, int n, const std::string& separator) {
        std::ostringstream ss;
        ss << name << separator << n;
        return ss.str();
    }

    std::string GridDescriptor::StripSuffix(const std::string& name, const std::string& separator) {
        return name.substr(0, name.find(separator));
    }

    bool GridDescriptor::Read(StreamReader* sr, const uint32_t file_version,
        std::string* err) {
        (void)file_version;

        unique_name_ = ReadString(sr);
        grid_name_ = StripSuffix(unique_name_);

        grid_type_ = ReadString(sr);

        // In order not to break backward compatibility with existing VDB files,
        // grids stored using 16-bit half floats are flagged by adding the following
        // suffix to the grid's type name on output.  The suffix is removed on input
        // and the grid's "save float as half" flag set accordingly.
        const char* HALF_FLOAT_TYPENAME_SUFFIX = "_HalfFloat";

        if (EndsWidth(grid_type_, HALF_FLOAT_TYPENAME_SUFFIX)) {
            save_float_as_half_ = true;
            // strip suffix
            std::string tmp =
                grid_type_.substr(0, grid_type_.find(HALF_FLOAT_TYPENAME_SUFFIX));
            grid_type_ = tmp;
        }

        // FIXME(syoyo): Currently only `Tree_float_5_4_3` type is supported.
        if (grid_type_.compare("Tree_float_5_4_3") != 0) {
            if (err) {
                (*err) = "Unsupported grid type: " + grid_type_;
            }
            return false;
        }

        std::cout << "grid_type = " << grid_type_ << std::endl;
        std::cout << "half = " << save_float_as_half_ << std::endl;

        {
            instance_parent_name_ = ReadString(sr);
            std::cout << "instance_parent_name = " << instance_parent_name_
                << std::endl;
        }

        // Create the grid of the type if it has been registered.
        // if (!GridBase::isRegistered(mGridType)) {
        //    OPENVDB_THROW(LookupError, "Cannot read grid." <<
        //        " Grid type " << mGridType << " is not registered.");
        //}
        // else
        // GridBase::Ptr grid = GridBase::createGrid(mGridType);
        // if (grid) grid->setSaveFloatAsHalf(mSaveFloatAsHalf);

        // Read in the offsets.
        sr->read8(&grid_byte_offset_);
        sr->read8(&block_byte_offset_);
        sr->read8(&end_byte_offset_);

        std::cout << "grid_byte_offset = " << grid_byte_offset_ << std::endl;
        std::cout << "block_byte_offset = " << block_byte_offset_ << std::endl;
        std::cout << "end_byte_offset = " << end_byte_offset_ << std::endl;

        return true;
    }

    static bool ReadMeta(StreamReader* sr) {
        // Read the number of metadata items.
        int count = 0;
        sr->read4(&count);

        if (count > 1024) {
            // Too many meta values.
            return false;
        }

        std::cout << "meta_count = " << count << std::endl;

        for (int i = 0; i < count; i++) {
            std::string name = ReadString(sr);

            // read typename string
            std::string type_name = ReadString(sr);

            std::cout << "meta[" << i << "] name = " << name
                << ", type_name = " << type_name << std::endl;

            if (type_name.compare("string") == 0) {
                std::string value = ReadString(sr);

                std::cout << "  value = " << value << std::endl;

            }
            else if (type_name.compare("vec3i") == 0) {
                int v[3];
                ReadMetaVec3i(sr, v);

                std::cout << "  value = " << v[0] << ", " << v[1] << ", " << v[2]
                    << std::endl;

            }
            else if (type_name.compare("vec3d") == 0) {
                double v[3];
                ReadMetaVec3d(sr, v);

                std::cout << "  value = " << v[0] << ", " << v[1] << ", " << v[2]
                    << std::endl;

            }
            else if (type_name.compare("bool") == 0) {
                bool b = ReadMetaBool(sr);

                std::cout << "  value = " << b << std::endl;

            }
            else if (type_name.compare("float") == 0) {
                float f = ReadMetaFloat(sr);

                std::cout << "  value = " << f << std::endl;

            }
            else if (type_name.compare("int64") == 0) {
                int64_t i64 = ReadMetaInt64(sr);

                std::cout << "  value = " << i64 << std::endl;

            }
            else {
                // Unknown metadata
                int num_bytes;
                sr->read4(&num_bytes);

                std::cout << "  unknown value. size = " << num_bytes << std::endl;

                std::vector<char> data;
                data.resize(size_t(num_bytes));
                sr->read(size_t(num_bytes), uint64_t(num_bytes),
                    reinterpret_cast<uint8_t*>(data.data()));
            }
        }

        return true;
    }

    static bool ReadGridDescriptors(StreamReader* sr, const uint32_t file_version,
        std::map<std::string, GridDescriptor>* gd_map) {
        // Read the number of grid descriptors.
        int count = 0;
        sr->read4(&count);

        std::cout << "grid descriptor counts = " << count << std::endl;

        for (int i = 0; i < count; ++i) {
            // Read the grid descriptor.
            GridDescriptor gd;
            std::string err;
            bool ret = gd.Read(sr, file_version, &err);
            if (!ret) {
                return false;
            }

            //  // Add the descriptor to the dictionary.
            (*gd_map)[gd.GridName()] = gd;

            // Move to the next descriptor.
            sr->seek_set(gd.EndByteOffset());
        }

        return true;
    }

    static bool ReadTransform(StreamReader* sr, std::string* err) {
        // Read the type name.
        std::string type = ReadString(sr);

        std::cout << "transform type = " << type << std::endl;

        double scale_values[3];
        double voxel_size[3];
        double scale_values_inverse[3];
        double inv_scale_squared[3];
        double inv_twice_scale[3];

        if ((type.compare("UniformScaleMap") == 0) ||
            (type.compare("UniformScaleTranslateMap") == 0)) {
            std::cout << "offt = " << sr->tell() << std::endl;

            scale_values[0] = scale_values[1] = scale_values[2] = 0.0;
            voxel_size[0] = voxel_size[1] = voxel_size[2] = 0.0;
            scale_values_inverse[0] = scale_values_inverse[1] =
                scale_values_inverse[2] = 0.0;
            inv_scale_squared[0] = inv_scale_squared[1] = inv_scale_squared[2] = 0.0;
            inv_twice_scale[0] = inv_twice_scale[1] = inv_twice_scale[2] = 0.0;

            ReadVec3d(sr, scale_values);
            ReadVec3d(sr, voxel_size);
            ReadVec3d(sr, scale_values_inverse);
            ReadVec3d(sr, inv_scale_squared);
            ReadVec3d(sr, inv_twice_scale);

            std::cout << "scale_values " << scale_values[0] << ", " << scale_values[1]
                << ", " << scale_values[2] << std::endl;
            std::cout << "voxel_size " << voxel_size[0] << ", " << voxel_size[1] << ", "
                << voxel_size[2] << std::endl;
            std::cout << "scale_value_sinverse " << scale_values_inverse[0] << ", "
                << scale_values_inverse[1] << ", " << scale_values_inverse[2]
                << std::endl;
            std::cout << "inv_scale_squared " << inv_scale_squared[0] << ", "
                << inv_scale_squared[1] << ", " << inv_scale_squared[2]
                << std::endl;
            std::cout << "inv_twice_scale " << inv_twice_scale[0] << ", "
                << inv_twice_scale[1] << ", " << inv_twice_scale[2] << std::endl;
        }
        else {
            if (err) {
                (*err) = "Transform type `" + type + "' is not supported.\n";
            }
            return false;
        }

        return true;
    }

    static uint32_t ReadGridCompression(StreamReader* sr, uint32_t file_version) {
        uint32_t compression = TINYVDB_COMPRESS_NONE;
        if (file_version >= TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
            sr->read4(&compression);
        }
        return compression;
    }

    static bool ReadGrid(StreamReader* sr, const uint32_t file_version,
        const bool half_precision, const GridDescriptor& gd,
        std::string* warn, std::string* err) {
        // read compression per grid(optional)
        uint32_t grid_compression = ReadGridCompression(sr, file_version);

        DeserializeParams params;
        params.file_version = file_version;
        params.compression_flags = grid_compression;
        params.half_precision = half_precision;

        // read meta
        if (!ReadMeta(sr)) {
            return false;
        }

        // read transform
        if (!ReadTransform(sr, err)) {
            return false;
        }

        // read topology
        if (!gd.IsInstance()) {
            GridLayoutInfo layout_info;

            // TODO(syoyo): Construct node hierarchy based on header description.
            NodeInfo root(NODE_TYPE_ROOT, VALUE_TYPE_FLOAT, 0);
            NodeInfo intermediate1(NODE_TYPE_INTERNAL, VALUE_TYPE_FLOAT, 5);
            NodeInfo intermediate2(NODE_TYPE_INTERNAL, VALUE_TYPE_FLOAT, 4);
            NodeInfo leaf(NODE_TYPE_LEAF, VALUE_TYPE_FLOAT, 3);

            layout_info.Add(root);
            layout_info.Add(intermediate1);
            layout_info.Add(intermediate2);
            layout_info.Add(leaf);

            RootNode root_node(layout_info);

            // TreeBase
            {
                int buffer_count;
                sr->read4(&buffer_count);
                if (buffer_count != 1) {
                    if (warn) {
                        (*warn) += "multi-buffer trees are no longer supported.";
                    }
                }
            }

            if (!root_node.ReadTopology(sr, /* level */ 0, params, warn, err)) {
                return false;
            }

            // TODO(syoyo): Consider bbox(ROI)
            if (!root_node.ReadBuffer(sr, /* level */ 0, params, warn, err)) {
                return false;
            }

            std::cout << "end = " << sr->tell() << std::endl;

        }
        else {
            // TODO
            TINYVDBIO_ASSERT(0);
        }

        // Move to grid position
        sr->seek_set(uint64_t(gd.GridByteOffset()));

        return true;
    }

    VDBStatus ParseVDBHeader(const std::string& filename, VDBHeader* header,
        std::string* err) {
        if (header == NULL) {
            if (err) {
                (*err) = "Invalid function arguments";
            }
            return TINYVDBIO_ERROR_INVALID_ARGUMENT;
        }

        // TODO(Syoyo): Load only header region.
        std::vector<uint8_t> data = read_file_binary(filename, err);
        if (data.empty()) {
            return TINYVDBIO_ERROR_INVALID_FILE;
        }

        VDBStatus status = ParseVDBHeader(data.data(), data.size(), header, err);
        return status;
    }

    static bool IsBigEndian(void) {
        union {
            uint32_t i;
            char c[4];
        } bint = { 0x01020304 };

        return bint.c[0] == 1;
    }

    VDBStatus ParseVDBHeader(const uint8_t* data, const size_t len,
        VDBHeader* header, std::string* err) {
        int64_t magic;

        // OpenVDB stores data in little endian manner(e.g. x86).
        // Swap bytes for big endian architecture(e.g. Power, SPARC)
        bool swap_endian = IsBigEndian();

        StreamReader sr(data, len, swap_endian);

        // [0:7] magic number
        if (!sr.read8(&magic)) {
            if (err) {
                (*err) += "Failed to read magic number.\n";
            }
            return TINYVDBIO_ERROR_INVALID_HEADER;
        }

        if (magic == kOPENVDB_MAGIC) {
            std::cout << "bingo!" << std::endl;
        }
        else {
            if (err) {
                (*err) += "Invalid magic number for VDB.\n";
            }
            return TINYVDBIO_ERROR_INVALID_HEADER;
        }

        // [8:11] version
        uint32_t file_version = 0;
        if (!sr.read4(&file_version)) {
            if (err) {
                (*err) += "Failed to read file version.\n";
            }
            return TINYVDBIO_ERROR_INVALID_HEADER;
        }

        std::cout << "File version: " << file_version << std::endl;

        if (file_version < TINYVDB_FILE_VERSION_SELECTIVE_COMPRESSION) {
            if (err) {
                (*err) =
                    "VDB file version earlier than "
                    "TINYVDB_FILE_VERSION_SELECTIVE_COMPRESSION(220) is not supported.";
            }
            return TINYVDBIO_ERROR_UNIMPLEMENTED;
        }

        // Read the library version numbers (not stored prior to file format version
        // 211).
        uint32_t major_version = 0;
        uint32_t minor_version = 0;
        if (file_version >= 211) {
            sr.read4(&major_version);
            std::cout << "major version : " << major_version << std::endl;
            sr.read4(&minor_version);
            std::cout << "minor version : " << minor_version << std::endl;
        }

        // Read the flag indicating whether the stream supports partial reading.
        // (Versions prior to 212 have no flag because they always supported partial
        // reading.)
        char has_grid_offsets = 0;
        if (file_version >= 212) {
            sr.read1(&has_grid_offsets);
            std::cout << "InputHasGridOffsets = "
                << (has_grid_offsets ? " yes " : " no ") << std::endl;
        }

        if (!has_grid_offsets) {
            // Unimplemened.
            if (err) {
                (*err) = "VDB withoput grid offset is not supported in TinyVDBIO.";
            }
            return TINYVDBIO_ERROR_UNIMPLEMENTED;
        }

        // 5) Read the flag that indicates whether data is compressed.
        //    (From version 222 on, compression information is stored per grid.)
        // mCompression = DEFAULT_COMPRESSION_FLAGS;
        // if (file_version < TINYVDB_FILE_VERSION_BLOSC_COMPRESSION) {
        //    // Prior to the introduction of Blosc, ZLIB was the default compression
        //    scheme. mCompression = (COMPRESS_ZIP | COMPRESS_ACTIVE_MASK);
        //}
        char is_compressed = 0;
        if (file_version >= TINYVDB_FILE_VERSION_SELECTIVE_COMPRESSION &&
            file_version < TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
            sr.read1(&is_compressed);
            std::cout << "Global Compression : " << (is_compressed != 0 ? "zip" : "none")
                << std::endl;
        }

        // 6) Read uuid.
        {
            // ASCII UUID = 32 chars + 4 hyphens('-') = 36 bytes.
            char uuid[36];
            sr.read(36, 36, reinterpret_cast<uint8_t*>(uuid));
            std::string uuid_string = std::string(uuid, 36);
            // TODO(syoyo): Store UUID somewhere.
            std::cout << "uuid ASCII: " << uuid_string << std::endl;
            header->uuid = std::string(uuid, 36);
        }

        header->file_version = file_version;
        header->major_version = major_version;
        header->minor_version = minor_version;
        // header->has_grid_offsets = has_grid_offsets;
        header->is_compressed = is_compressed;
        header->offset_to_data = sr.tell();

        return TINYVDBIO_SUCCESS;
    }

    VDBStatus ReadGridDescriptors(const std::string& filename,
        const VDBHeader& header,
        std::map<std::string, GridDescriptor>* gd_map,
        std::string* err) {
        std::vector<uint8_t> data = read_file_binary(filename, err);
        if (data.empty()) {
            return TINYVDBIO_ERROR_INVALID_FILE;
        }

        VDBStatus status =
            ReadGridDescriptors(data.data(), data.size(), header, gd_map, err);
        return status;
    }

    VDBStatus ReadGridDescriptors(const uint8_t* data, const size_t data_len,
        const VDBHeader& header,
        std::map<std::string, GridDescriptor>* gd_map,
        std::string* err) {
        bool swap_endian = IsBigEndian();
        StreamReader sr(data, data_len, swap_endian);

        if (!sr.seek_set(header.offset_to_data)) {
            if (err) {
                (*err) += "Failed to seek into data.\n";
            }
            return TINYVDBIO_ERROR_INVALID_DATA;
        }

        // Read meta data.
        {
            bool ret = ReadMeta(&sr);
            std::cout << "meta: " << ret << std::endl;
        }

        if (!ReadGridDescriptors(&sr, header.file_version, gd_map)) {
            if (err) {
                (*err) += "Failed to read grid descriptors.\n";
            }
            return TINYVDBIO_ERROR_INVALID_DATA;
        }

        return TINYVDBIO_SUCCESS;
    }

    VDBStatus ReadGrids(const std::string& filename, const VDBHeader& header,
        const std::map<std::string, GridDescriptor>& gd_map,
        std::string* warn, std::string* err) {
        std::vector<uint8_t> data = read_file_binary(filename, err);
        if (data.empty()) {
            return TINYVDBIO_ERROR_INVALID_FILE;
        }

        VDBStatus status =
            ReadGrids(data.data(), data.size(), header, gd_map, warn, err);
        return status;
    }

    VDBStatus ReadGrids(const uint8_t* data, const size_t data_len,
        const VDBHeader& header,
        const std::map<std::string, GridDescriptor>& gd_map,
        std::string* warn, std::string* err) {
        bool swap_endian = IsBigEndian();
        StreamReader sr(data, data_len, swap_endian);

        std::cout << "AAA: num_grids = " << gd_map.size() << "\n";
        for (const auto& it : gd_map) {
            const GridDescriptor& gd = it.second;

            sr.seek_set(gd.GridByteOffset());

            if (!ReadGrid(&sr, header.file_version, header.half_precision, gd, warn,
                err)) {
                if (err) {
                    (*err) += "Failed to read Grid data.\n";
                }
                return TINYVDBIO_ERROR_INVALID_DATA;
            }
        }

        return TINYVDBIO_SUCCESS;
    }

    static bool WriteVDBHeader(std::ostream& os) {
        // [0:7] magic number
        int64_t magic = kOPENVDB_MAGIC;
        os.write(reinterpret_cast<char*>(&magic), 8);

        // [8:11] version
        uint32_t file_version = kTINYVDB_FILE_VERSION;
        os.write(reinterpret_cast<char*>(&file_version), sizeof(uint32_t));

#if 0  // TODO(syoyo): Implement

        std::cout << "File version: " << file_version << std::endl;

        // Read the library version numbers (not stored prior to file format version
        // 211).
        if (file_version >= 211) {
            uint32_t version;
            ifs.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
            std::cout << "major version : " << version << std::endl;
            ifs.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
            std::cout << "minor version : " << version << std::endl;
        }

        // Read the flag indicating whether the stream supports partial reading.
        // (Versions prior to 212 have no flag because they always supported partial
        // reading.)
        char has_grid_offsets = 0;
        if (file_version >= 212) {
            ifs.read(&has_grid_offsets, sizeof(char));
            std::cout << "InputHasGridOffsets = " << (has_grid_offsets ? " yes " : " no ")
                << std::endl;
        }

        // 5) Read the flag that indicates whether data is compressed.
        //    (From version 222 on, compression information is stored per grid.)
        // mCompression = DEFAULT_COMPRESSION_FLAGS;
        // if (file_version < TINYVDB_FILE_VERSION_BLOSC_COMPRESSION) {
        //    // Prior to the introduction of Blosc, ZLIB was the default compression
        //    scheme. mCompression = (COMPRESS_ZIP | COMPRESS_ACTIVE_MASK);
        //}
        char isCompressed = 0;
        if (file_version >= TINYVDB_FILE_VERSION_SELECTIVE_COMPRESSION &&
            file_version < TINYVDB_FILE_VERSION_NODE_MASK_COMPRESSION) {
            ifs.read(&isCompressed, sizeof(char));
            std::cout << "Compression : " << (isCompressed != 0 ? "zip" : "none")
                << std::endl;
        }

        // 6) Read the 16-byte(128-bit) uuid.
        if (file_version >= TINYVDB_FILE_VERSION_BOOST_UUID) {
            // ASCII UUID = 32 chars + 4 '-''s = 36 bytes.
            char uuid[36];
            ifs.read(uuid, 36);
            // TODO(syoyo): Store UUID somewhere.
            std::cout << "uuid: " << uuid << std::endl;
        }
        else {
            char uuid[16];
            ifs.read(uuid, 16);
            // TODO(syoyo): Store UUID somewhere.
            std::cout << "uuid: " << uuid << std::endl;
        }

        {
            bool ret = ReadMeta(ifs);
            std::cout << "meta: " << ret << std::endl;
        }

        if (has_grid_offsets) {
            ReadGridDescriptors(ifs);
        }
        else {
        }
#endif

        return true;
    }

    bool SaveVDB(const std::string& filename, std::string* err) {
        std::ofstream os(filename.c_str(), std::ifstream::binary);

        if (!os) {
            if (err) {
                (*err) = "Failed to open a file to write: " + filename;
            }
            return false;
        }

        WriteVDBHeader(os);
        // if filemane

        return true;
    }

    void VoxelTree::BuildTree(const InternalOrLeafNode& root, int depth) {
        (void)root;
        (void)depth;
    }

    bool VoxelTree::Build(const RootNode& root, std::string* err) {
        nodes_.clear();

        const GridLayoutInfo& grid_layout_info = root.GetGridLayoutInfo();
        (void)grid_layout_info;

        // toplevel.
        // Usually divided into 2*2*2 region for 5_4_3 tree configuration
        // (each region has 4096^3 voxel size)
        if (root.GetChildBounds().size() != root.GetChildNodes().size()) {
            if (err) {
                (*err) = "Invalid RootNode.\n";
            }
            return false;
        }

        // root node
        VoxelNode node;

        nodes_.push_back(node);

        Boundsi root_bounds;
        for (size_t i = 0; i < root.GetChildNodes().size(); i++) {
            // TODO(syoyo): Check overlap
            root_bounds = Boundsi::Union(root_bounds, root.GetChildBounds()[i]);
            BuildTree(root.GetChildNodes()[i], 0);
        }

        std::cout << root_bounds << std::endl;

        valid_ = true;
        return true;
    }

}  // namespace tinyvdb

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif  // TINYVDBIO_IMPLEMENTATION