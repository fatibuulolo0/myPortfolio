import { withBase } from "./utils/helpers";

export type Image = {
    src: string;
    alt?: string;
    caption?: string;
};

export type Link = {
    text: string;
    href: string;
};

export type Hero = {
    eyebrowText?: string;
    title?: string;
    text?: string;
    image?: Image;
    actions?: Link[];
};

export type About = {
    title?: string;
    text?: string;
};

export type Blog = {
    description?: string;
};

export type ContactInfo = {
    title?: string;
    text?: string;
    email?: {
        text?: string;
        href?: string;
        email?: string;
    };
    socialProfiles?: {
        text?: string;
        href?: string;
    }[];
};

export type Subscribe = {
    title?: string;
    text?: string;
    formUrl: string;
};

export type SiteConfig = {
    website: string;
    logo?: Image;
    title: string;
    description: string;
    image?: Image;
    headerNavLinks?: Link[];
    footerNavLinks?: Link[];
    socialLinks?: Link[];
    hero?: Hero;
    about?: About;
    contactInfo?: ContactInfo;
    subscribe?: Subscribe;
    blog?: Blog;
    postsPerPage?: number;
    recentPostLimit: number;
    projectsPerPage?: number;
};

const siteConfig: SiteConfig = {
    website: 'https://example.com',
    title: 'Space Ahead',
    description: 'A minimal space-inspired personal blog template built with Astro.js and Tailwind CSS, by Siddhesh Thadeshwar',
    image: {
        src: '/space-ahead-preview.jpeg',
        alt: 'Space Ahead ✨ - A minimal space-inspired personal blog template, created by Siddhesh Thadeshwar.'
    },
    headerNavLinks: [
        {
            text: 'Home',
            href: withBase('/')
        },
        {
            text: 'Skills',
            href: withBase('/blog')
        },
        {
            text: 'Projects',
            href: withBase('/tags')
        },
        {
            text: 'About',
            href: withBase('/about')
        },
        {
            text: 'Contact',
            href: withBase('/contact')
        }
    ],   
    socialLinks: [
        {
            text: 'GitHub',
            href: 'https://github.com/FatiBuuloloo'
        },
        {
            text: 'Instagram',
            href: 'https://www.instagram.com/fatibuulolo/'
        },
        {
            text: 'LinkedIn',
            href: 'https://www.linkedin.com/in/fati-buulolo-7a9236391/'
        }
    ],
    hero: {
        eyebrowText: 'Galaxy of Adventures',
        title: 'Fati Buulolo',
        text: "Written by Astro-naut Sid, a space explorer at Beyond Earth.",
        image: {
            src: '/assets/images/pixeltrue-space-discovery.svg',
            alt: 'A person sitting at a desk in front of a computer'
        },
        actions: [
            {
                text: 'Read Now',
                href: withBase('/blog')
            }          
        ]
    },
    about: {
        title: 'About',
        text: 'Space Ahead is a blog about space exploration and travel. It is written by Astro-naut Sid, a space explorer at Beyond Earth. Sid is known for his love of adventure and his insatiable curiosity about the universe. He has explored countless planets, discovered new life forms, and made friends with aliens along the way. 🚀',
    },
    contactInfo: {
        title: 'Contact',
        text: "Hi! Whether you have a question, a suggestion, or just want to share your thoughts, I'm all ears. Feel free to get in touch through any of the methods below:",
        email: {
            text: "Drop me an email and I’ll do my best to respond as soon as possible.",
            href: "mailto:fatibuulolo2@yahoo.com",
            email: "yahoo.com"
        },
        socialProfiles: [
            {
                text: "LinkedIn",
                href: "https://www.linkedin.com/in/fati-buulolo-7a9236391/"
            },
            {
                text: "GitHub",
                href: "https://github.com/FatiBuuloloo"
            }
        ]
    },    
    blog: {
        description: "Read about my space adventures, explorations and the aliens I've met on my journeys."
    },
    postsPerPage: 2,
    recentPostLimit: 3
};

export default siteConfig;
