/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>

#include "scheduler/scheduler_types.h"

namespace scheduler_protocol_export {

// Field names and string values are schema literals. Device values are numeric only.
class JsonObject {
public:
    explicit JsonObject(std::ostream &stream) :
        stream_(stream) {
        stream_ << '{';
    }
    ~JsonObject() { stream_ << '}'; }
    JsonObject(const JsonObject &) = delete;
    JsonObject &operator=(const JsonObject &) = delete;

    std::ostream &field(const char *name) {
        if (!first_) stream_ << ',';
        first_ = false;
        return stream_ << '"' << name << "\":";
    }
    template <class T>
    void number(const char *name, T value) {
        field(name) << +value;
    }
    void string(const char *name, const char *value) { field(name) << '"' << value << '"'; }
    void stamp(const char *name, const SchedulerProtocolStamp &value) {
        field(name) << '[' << value.cycles << ',' << value.sequence << ']';
    }
    template <class T>
    void numbers(const char *name, const T *values, size_t count) {
        field(name) << '[';
        for (size_t index = 0; index < count; ++index) {
            if (index != 0) stream_ << ',';
            stream_ << +values[index];
        }
        stream_ << ']';
    }

private:
    std::ostream &stream_;
    bool first_{true};
};

class Image {
public:
    Image(const void *base, size_t size) :
        base_(static_cast<const uint8_t *>(base)),
        size_(size) {
        if (base == nullptr || reinterpret_cast<uintptr_t>(base) % SCHEDULER_STATE_ALIGNMENT != 0)
            throw std::invalid_argument("invalid protocol image base");
    }
    template <class T>
    const T *array(uint64_t offset, uint64_t count) const {
        if (count == 0) return nullptr;
        if (offset % alignof(T) != 0 || offset > size_ || count > (size_ - offset) / sizeof(T))
            throw std::invalid_argument("protocol image region out of bounds");
        return reinterpret_cast<const T *>(base_ + offset);
    }

private:
    const uint8_t *base_;
    size_t size_;
};

template <class T>
inline bool populated(const T &record) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&record);
    return std::any_of(bytes, bytes + sizeof(T), [](uint8_t value) {
        return value != 0;
    });
}

template <class T, class Emit>
inline void sparse(JsonObject &root, const char *name, const T *records, uint64_t count, Emit emit) {
    auto &out = root.field(name);
    out << '[';
    bool first = true;
    for (uint64_t index = 0; index < count; ++index) {
        if (!populated(records[index])) continue;
        if (!first) out << ',';
        first = false;
        JsonObject row(out);
        row.number("index", index);
        emit(row, records[index]);
    }
    out << ']';
}

}  // namespace scheduler_protocol_export
