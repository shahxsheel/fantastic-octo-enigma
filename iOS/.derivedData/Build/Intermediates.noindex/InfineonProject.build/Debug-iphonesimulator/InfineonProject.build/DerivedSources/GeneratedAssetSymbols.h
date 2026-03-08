#import <Foundation/Foundation.h>

#if __has_attribute(swift_private)
#define AC_SWIFT_PRIVATE __attribute__((swift_private))
#else
#define AC_SWIFT_PRIVATE
#endif

/// The "benji" asset catalog image resource.
static NSString * const ACImageNameBenji AC_SWIFT_PRIVATE = @"benji";

/// The "modelY" asset catalog image resource.
static NSString * const ACImageNameModelY AC_SWIFT_PRIVATE = @"modelY";

/// The "sampleAppScreenshot" asset catalog image resource.
static NSString * const ACImageNameSampleAppScreenshot AC_SWIFT_PRIVATE = @"sampleAppScreenshot";

#undef AC_SWIFT_PRIVATE
